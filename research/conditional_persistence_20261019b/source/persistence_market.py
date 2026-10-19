"""Frozen conditional-persistence market model and training-only references.

Implementation role: prior_audit. This code requires independent review by a
different researcher; its author is no longer the independent model auditor.
No market data are read and no fitting occurs at import time.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp, ndtr, softmax
import torch
from torch import nn
from torch.nn import functional as F

try:
    from .persistence import PI, Q, HORIZONS, SHIFT, propagate, abs_normal, pair_matrix
except ImportError:
    from persistence import PI, Q, HORIZONS, SHIFT, propagate, abs_normal, pair_matrix


ARMS = {
    "GRU16": ("conditional_joint005", "constant_joint005",
              "conditional_endpoint", "constant_endpoint",
              "scale_capacity_joint005", "independent_horizon_endpoint"),
    "MLP16": ("conditional_joint005", "constant_joint005"),
}
SEEDS = (17, 41, 89)
EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = .002
WEIGHT_DECAY = .001
SQRT2PI = math.sqrt(2*math.pi)
SQRT2_OVER_PI = math.sqrt(2/math.pi)


def finite(value, label):
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if not np.all(np.isfinite(array)):
        raise FloatingPointError(f"Nonfinite {label}; no clipping/fallback is allowed")


def reference_mode(arm):
    return "joint005" if arm.endswith("joint005") else "endpoint"


def _numpy_crps(y, weights, sigmas):
    z = y[:, None]/sigmas
    first = (sigmas*SQRT2_OVER_PI*np.exp(-.5*z*z)
             + y[:, None]*(2*ndtr(z)-1)) @ weights
    pair = SQRT2_OVER_PI*np.sqrt(sigmas[:, None]**2+sigmas[None, :]**2)
    return first-.5*weights@pair@weights


def fit_h1_reference(train_y):
    """One fixed-budget fit of pi/f0, using training h1 CRPS only."""
    y = np.asarray(train_y, dtype=np.float64)
    if y.ndim == 2:
        y = y[:, 0]
    if y.ndim != 1 or not len(y):
        raise ValueError("Training h1 target vector required")
    finite(y, "H1 training targets")
    initial = np.r_[np.log(PI[:2]/PI[2]), 1.]

    def objective(parameters):
        pi = softmax(np.r_[parameters[:2], 0.])
        sd = parameters[2]*np.sqrt(Q/(pi@Q))
        answer = float(_numpy_crps(y, pi, sd).mean())
        finite(answer, "H1 objective")
        return answer

    started = time.perf_counter()
    initial_fun = objective(initial)
    result = minimize(objective, initial, method="L-BFGS-B",
                      bounds=[(-6.,6.),(-6.,6.),(.25,2.5)],
                      options={"maxiter":300})
    parameters = np.asarray(result.x, dtype=float)
    finite(parameters, "H1 reference solution")
    if np.any(np.abs(parameters[:2]) > 6.+1e-10) or not .25-1e-10 <= parameters[2] <= 2.5+1e-10:
        raise FloatingPointError("H1 reference outside fixed bounds")
    pi = softmax(np.r_[parameters[:2], 0.])
    f0 = float(parameters[2])
    final_fun = objective(parameters)
    # A finite budget-ended solution is retained with its actual status;
    # no success-dependent retry, changed bounds, or alternative start.
    receipt = {
        "estimator":"H1_global_GMM_CRPS", "training_origins":len(y),
        "objective":"mean training h1 CRPS only", "initial_pi":PI.tolist(),
        "initial_f0":1., "pi":pi.tolist(), "f0":f0,
        "success":bool(result.success), "status":int(result.status),
        "message":str(result.message), "nit":int(result.nit),
        "nfev":int(result.nfev), "initial_fun":initial_fun,
        "fun":final_fun, "optimizer_reported_fun":float(result.fun),
        "gradient_norm":float(np.linalg.norm(result.jac)),
        "maxiter":300, "retained_finite_budget_solution":True,
        "restarts":0, "elapsed_seconds":time.perf_counter()-started,
    }
    finite(result.fun, "H1 reported objective")
    finite(result.jac, "H1 reported gradient")
    return pi, f0, receipt


def _reference_log_emissions(path, pi, f0):
    sd = f0*np.sqrt(Q/(pi@Q))
    z = path[:, :, None]/sd
    result = -.5*z*z-np.log(sd)-.5*np.log(2*np.pi)
    finite(result, "reference log emissions")
    return result


def _numpy_path_nll(log_emissions, rho, pi):
    """Direct log-domain forward algorithm for a scalar reference rho."""
    log_pi = np.log(pi)
    log_alpha = np.broadcast_to(log_pi, log_emissions.shape[::2]).copy()
    nll = np.zeros(len(log_emissions))
    for j in range(log_emissions.shape[1]):
        joint = log_alpha+log_emissions[:, j, :]
        normalizer = logsumexp(joint, axis=-1)
        nll -= normalizer
        log_posterior = joint-normalizer[:, None]
        log_alpha = np.logaddexp(np.log(rho)+log_posterior,
                                np.log1p(-rho)+log_pi)
    answer = nll/log_emissions.shape[1]
    finite(answer, "reference path NLL")
    return answer


def fit_rho_reference(train_path, train_y, pi, f0, auxiliary):
    """21 fixed grid evaluations plus one bounded scalar refinement."""
    path = np.asarray(train_path, dtype=np.float64)
    y = np.asarray(train_y, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 20 or y.shape != (len(path),3):
        raise ValueError("Expected common-origin [n,20] path and [n,3] target")
    finite(path, "reference paths")
    skeleton = propagate([.5], pi=pi)
    first = {h: abs_normal(y[:,index], f0*skeleton[h][1]).mean(0)
             for index,h in enumerate(HORIZONS)}
    pairs = {h:f0*pair_matrix(skeleton[h][1]) for h in HORIZONS}
    log_emissions = _reference_log_emissions(path, pi, f0)
    calls = 0

    def objective(rho):
        nonlocal calls
        calls += 1
        distributions = propagate([rho],pi=pi)
        endpoint = np.mean([float(distributions[h][0][0]@first[h]
                         -.5*distributions[h][0][0]@pairs[h]@distributions[h][0][0])
                            for h in HORIZONS])
        path_loss = float(_numpy_path_nll(log_emissions,rho,pi).mean()) if auxiliary else 0.
        value = float(endpoint+auxiliary*path_loss)
        finite(value, "constant-rho objective")
        return value

    started = time.perf_counter()
    grid = np.linspace(.001,.999,21)
    losses = np.asarray([objective(float(rho)) for rho in grid])
    best = int(np.argmin(losses))
    bounds = (float(grid[max(0,best-1)]),float(grid[min(20,best+1)]))
    result = minimize_scalar(objective,bounds=bounds,method="bounded",
                             options={"xatol":1e-7,"maxiter":100})
    finite([result.x,result.fun], "constant-rho refinement")
    candidates = [(float(grid[best]),float(losses[best]),"grid"),
                  (float(result.x),float(result.fun),"refinement"),
                  (float(grid[0]),float(losses[0]),"lower_endpoint"),
                  (float(grid[-1]),float(losses[-1]),"upper_endpoint")]
    rho,loss,selected = min(candidates,key=lambda value:(value[1],value[0]))
    if not .001 <= rho <= .999:
        raise FloatingPointError("Constant rho outside fixed bounds")
    receipt = {
        "estimator":"constant_rho_joint005" if auxiliary else "constant_rho_endpoint",
        "rho":rho,"auxiliary":auxiliary,"training_origins":len(path),
        "fun":loss,"selected_candidate":selected,"grid":grid.tolist(),
        "grid_objectives":losses.tolist(),"grid_best_index":best,
        "refinement_bounds":list(bounds),"success":bool(result.success),
        "status":int(result.status),"message":str(result.message),
        "nit":int(result.nit),"nfev":int(result.nfev),
        "refinement_fun":float(result.fun),"initial_fun":float(losses[10]),
        "initial_rho":float(grid[10]),"objective_calls":calls,
        "restarts":0,"elapsed_seconds":time.perf_counter()-started,
    }
    return rho,receipt


def fit_references(train_path,train_y):
    pi,f0,h1 = fit_h1_reference(train_y)
    rho_endpoint,endpoint = fit_rho_reference(train_path,train_y,pi,f0,0.)
    rho_joint,joint = fit_rho_reference(train_path,train_y,pi,f0,.05)
    return {"pi":pi.tolist(),"f0":f0,
            "rho_endpoint":rho_endpoint,"rho_joint005":rho_joint,
            "statistical_fits":3,"receipts":[h1,endpoint,joint]}


class PersistenceMarketModel(nn.Module):
    """Float32 encoder/linear logits; float64 transforms, DP, and densities."""
    def __init__(self,backbone,arm,reference):
        super().__init__()
        if backbone not in ARMS or arm not in ARMS[backbone]:
            raise ValueError("Unregistered backbone/arm")
        self.backbone_name,self.arm = backbone,arm
        self.auxiliary = .05 if arm.endswith("joint005") else 0.
        self.independent = arm == "independent_horizon_endpoint"
        self.capacity = arm == "scale_capacity_joint005"
        self.conditional = arm.startswith("conditional_")
        if backbone == "GRU16":
            self.encoder = nn.GRU(3,16,batch_first=True)
        else:
            self.encoder = nn.Sequential(nn.Flatten(),nn.Linear(60,16),nn.Tanh(),
                                         nn.Linear(16,16),nn.Tanh())
        outputs = 3 if self.independent else 2 if (self.conditional or self.capacity) else 1
        self.head = nn.Linear(16,outputs)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        pi = torch.as_tensor(reference["pi"],dtype=torch.float64)
        f0 = float(reference["f0"])
        rho0 = float(reference["rho_"+reference_mode(arm)])
        if not np.isfinite(f0) or f0 <= 0 or not 0 < rho0 < 1:
            raise ValueError("Invalid fixed reference")
        self.register_buffer("pi",pi)
        self.register_buffer("f0",torch.tensor(f0,dtype=torch.float64))
        self.register_buffer("base_rho",torch.tensor(rho0,dtype=torch.float64))
        self.register_buffer("base_logit",torch.tensor(math.log(rho0)-math.log1p(-rho0),dtype=torch.float64))
        self.register_buffer("state_sigma",torch.sqrt(torch.as_tensor(Q,dtype=torch.float64)/(pi@torch.as_tensor(Q,dtype=torch.float64))))
        skeleton = propagate([.5],pi=pi.numpy())
        for h in HORIZONS:
            sd = torch.as_tensor(skeleton[h][1],dtype=torch.float64)
            self.register_buffer(f"sigma_{h}",sd)
            self.register_buffer(f"pair_{h}",SQRT2_OVER_PI*torch.sqrt(sd[:,None]**2+sd[None,:]**2))

    def forecast_parameters(self,x):
        if x.ndim != 3 or x.shape[1:] != (20,3):
            raise ValueError("Expected the common [batch,20,3] feature tensor")
        if self.backbone_name == "GRU16":
            encoded,_ = self.encoder(x.to(dtype=torch.float32))
            encoded = encoded[:,-1,:]
        else:
            encoded = self.encoder(x.to(dtype=torch.float32))
        raw = self.head(encoded).to(dtype=torch.float64)
        scale_index = raw[:,0]+.5*torch.tanh(raw[:,1]) if self.capacity else raw[:,0]
        f = self.f0*(.5+torch.sigmoid(scale_index))
        if self.independent:
            rho = torch.sigmoid(self.base_logit+raw[:,1:3])
        elif self.conditional:
            shared = torch.sigmoid(self.base_logit+raw[:,1])
            rho = torch.stack((shared,shared),dim=-1)
        else:
            rho = self.base_rho.expand(len(x),2)
        # Keep stable log mixing probabilities even when sigmoid rounds to 0/1.
        if self.conditional:
            logit = self.base_logit+raw[:,1]
        elif self.independent:
            logit = None
        else:
            logit = self.base_logit.expand(len(x))
        return {"f":f,"rho":rho,"logit":logit}

    def weights(self,rho,horizons=HORIZONS):
        v = torch.stack([F.pad(torch.ones_like(rho)[:,None]*self.pi[k],(offset,5-offset))
                         for k,offset in enumerate(SHIFT)],dim=1)
        result = {}
        for h in range(1,max(horizons)+1):
            if h in horizons:
                result[h] = v.sum(1)
            if h == max(horizons):
                break
            total = v.sum(1)
            v = torch.stack([F.pad(rho[:,None]*v[:,k,:]
                            +(1-rho[:,None])*self.pi[k]*total,(offset,5-offset))
                             for k,offset in enumerate(SHIFT)],dim=1)
        return result

    def distributions(self,parameters):
        rho,f = parameters["rho"],parameters["f"]
        if self.independent:
            # Separate h5/h20 kernels; no assertion of common path consistency.
            both = self.weights(torch.cat((rho[:,0],rho[:,1])),horizons=HORIZONS)
            n = len(rho)
            weights = {1:both[1][:n],5:both[5][:n],20:both[20][n:]}
        else:
            weights = self.weights(rho[:,0])
        return {h:(weights[h],f[:,None]*getattr(self,f"sigma_{h}")[None,:])
                for h in HORIZONS}

    def loss_components(self,x,y,path,*,include_path=None):
        p = self.forecast_parameters(x)
        distribution = self.distributions(p)
        y,path = y.to(torch.float64),path.to(torch.float64)
        crps_h = []
        for index,h in enumerate(HORIZONS):
            w,sd = distribution[h]
            target = y[:,index,None]
            z = target/sd
            first = (sd*SQRT2_OVER_PI*torch.exp(-.5*z*z)
                     +target*torch.erf(z/math.sqrt(2)))
            pair = getattr(self,f"pair_{h}")
            value = (w*first).sum(-1)-.5*p["f"]*torch.einsum("bi,ij,bj->b",w,pair,w)
            crps_h.append(value)
        crps_h = torch.stack(crps_h,dim=-1)
        endpoint = crps_h.mean(-1)
        calculate_path = self.auxiliary > 0 if include_path is None else include_path
        if calculate_path and self.independent:
            raise ValueError("Independent-horizon model has no shared path NLL")
        if calculate_path:
            sd = p["f"][:,None]*self.state_sigma[None,:]
            log_emissions = -.5*(path[:,:,None]/sd[:,None,:])**2-torch.log(sd[:,None,:])-.5*math.log(2*math.pi)
            logpi = torch.log(self.pi)
            log_alpha = logpi.expand(len(x),3)
            nll = torch.zeros(len(x),dtype=torch.float64,device=x.device)
            log_stay = F.logsigmoid(p["logit"])[:,None]
            log_reset = F.logsigmoid(-p["logit"])[:,None]
            for step in range(path.shape[1]):
                joint = log_alpha+log_emissions[:,step,:]
                normalizer = torch.logsumexp(joint,dim=-1)
                nll = nll-normalizer
                posterior = joint-normalizer[:,None]
                log_alpha = torch.logaddexp(log_stay+posterior,log_reset+logpi)
            nll = nll/path.shape[1]
        else:
            nll = torch.zeros_like(endpoint)
        objective = endpoint+self.auxiliary*nll
        return {"objective":objective,"endpoint":endpoint,"crps_h":crps_h,
                "path_nll":nll,**p}


def role_tensors(arrays,role):
    return (torch.as_tensor(arrays[f"{role}_X"],dtype=torch.float32),
            torch.as_tensor(arrays[f"{role}_y"],dtype=torch.float64),
            torch.as_tensor(arrays[f"{role}_future_normalized"],dtype=torch.float64))


@torch.no_grad()
def evaluate_loss(model,tensors,batch_size=128):
    model.eval()
    x,y,path = tensors
    totals = np.zeros(6,dtype=float)
    for first in range(0,len(x),batch_size):
        values = model.loss_components(x[first:first+batch_size],y[first:first+batch_size],
                                       path[first:first+batch_size])
        block = torch.cat((values["objective"][:,None],values["endpoint"][:,None],
                           values["path_nll"][:,None],values["crps_h"]),dim=1)
        finite(block,"evaluation loss")
        totals += block.sum(0).cpu().numpy()
    totals /= len(x)
    return {"objective":float(totals[0]),"endpoint_CRPS":float(totals[1]),
            "path_NLL":float(totals[2]) if model.auxiliary else None,
            "CRPS_h":dict(zip(map(str,HORIZONS),totals[3:].tolist()))}


def fit_neural(arrays,reference,backbone,arm,seed,*,epochs=EPOCHS,progress=None):
    """All epochs run; validation chooses earliest strict improvement."""
    started = time.perf_counter()
    torch.manual_seed(seed)
    model = PersistenceMarketModel(backbone,arm,reference)
    train,validation = role_tensors(arrays,"train"),role_tensors(arrays,"validation")
    optimizer = torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    order_generator = np.random.default_rng(seed)
    initial_parameters = model.forecast_parameters(train[0][:min(8,len(train[0]))])
    rho0 = reference["rho_"+reference_mode(arm)]
    initial_f_error = float(torch.max(torch.abs(initial_parameters["f"]-reference["f0"])).detach())
    initial_rho_error = float(torch.max(torch.abs(initial_parameters["rho"]-rho0)).detach())
    if initial_f_error > 1e-13 or initial_rho_error > 1e-13:
        raise AssertionError("Zero-head reference initialization failed")
    history = [{"epoch":0,"training":evaluate_loss(model,train),
                "validation":evaluate_loss(model,validation),"optimizer_steps":0,
                "max_gradient_norm_before_clip":None}]
    best_score = history[0]["validation"]["endpoint_CRPS"]
    selected_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
    steps = 0
    for epoch in range(1,epochs+1):
        model.train()
        order = order_generator.permutation(len(train[0]))
        maximum_norm = 0.
        batch_objective_sum = 0.
        for first in range(0,len(order),BATCH_SIZE):
            ids = torch.as_tensor(order[first:first+BATCH_SIZE],dtype=torch.long)
            values = model.loss_components(train[0][ids],train[1][ids],train[2][ids])
            loss = values["objective"].mean()
            finite(loss,"training objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm_value = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            maximum_norm = max(maximum_norm,float(norm_value))
            optimizer.step()
            for parameter in model.parameters():
                finite(parameter,"trained model parameter")
            steps += 1
            batch_objective_sum += float(loss.detach())*len(ids)
        row = {"epoch":epoch,"training":evaluate_loss(model,train),
               "validation":evaluate_loss(model,validation),"optimizer_steps":steps,
               "minibatch_preupdate_objective":batch_objective_sum/len(order),
               "max_gradient_norm_before_clip":maximum_norm}
        history.append(row)
        score = row["validation"]["endpoint_CRPS"]
        if score < best_score:
            selected_epoch,best_score = epoch,score
            best_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
        if progress is not None:
            progress(epoch,steps,selected_epoch)
    model.load_state_dict(best_state)
    model.selected_optimizer_state = best_optimizer_state
    model.final_optimizer_state = copy.deepcopy(optimizer.state_dict())
    model.final_minibatch_generator_state = order_generator.bit_generator.state
    receipt = {"backbone":backbone,"arm":arm,"seed":seed,
               "train_origins":len(train[0]),"validation_origins":len(validation[0]),
               "epochs_run":epochs,"optimizer_steps":steps,"batch_size":BATCH_SIZE,
               "selected_epoch":selected_epoch,"selected_validation_CRPS":best_score,
               "epoch_zero_selected":selected_epoch == 0,
               "parameter_count":sum(p.numel() for p in model.parameters()),
               "encoder_dtype":"float32","linear_logit_projection_dtype":"float32",
               "probability_transform_DP_density_dtype":"float64",
               "reference_mode":reference_mode(arm),"auxiliary_weight":model.auxiliary,
               "initial_f_absolute_error":initial_f_error,
               "initial_rho_absolute_error":initial_rho_error,
               "history":history,"elapsed_seconds":time.perf_counter()-started,
               "training_complete_before_validation_selected_state_reload":True,
               "test_used_for_selection":False}
    return model,receipt


@torch.no_grad()
def predict_parameters(model,x,batch_size=128):
    """Small training-side archive; complete mixtures remain replayable."""
    model.eval()
    result = {"f":[],"rho_h5":[],"rho_h20":[]}
    tensor = torch.as_tensor(x,dtype=torch.float32)
    for first in range(0,len(tensor),batch_size):
        p = model.forecast_parameters(tensor[first:first+batch_size])
        result["f"].append(p["f"].cpu().numpy())
        result["rho_h5"].append(p["rho"][:,0].cpu().numpy())
        result["rho_h20"].append(p["rho"][:,1].cpu().numpy())
    return {key:np.concatenate(value) for key,value in result.items()}


@torch.no_grad()
def predict(model,x,batch_size=128):
    model.eval()
    result = {"f":[],"rho_h5":[],"rho_h20":[]}
    for h in HORIZONS:
        result[f"weights_h{h}"]=[]
        result[f"sigmas_h{h}"]=[]
    tensor = torch.as_tensor(x,dtype=torch.float32)
    for first in range(0,len(tensor),batch_size):
        parameters = model.forecast_parameters(tensor[first:first+batch_size])
        distribution = model.distributions(parameters)
        result["f"].append(parameters["f"].cpu().numpy())
        result["rho_h5"].append(parameters["rho"][:,0].cpu().numpy())
        result["rho_h20"].append(parameters["rho"][:,1].cpu().numpy())
        for h,(weights,sigmas) in distribution.items():
            finite(weights,"prediction weights")
            finite(sigmas,"prediction sigmas")
            if torch.any(weights < 0) or torch.max(torch.abs(weights.sum(-1)-1)) > 2e-12:
                raise FloatingPointError("Invalid predictive mixture mass")
            result[f"weights_h{h}"].append(weights.cpu().numpy())
            result[f"sigmas_h{h}"].append(sigmas.cpu().numpy())
    return {key:np.concatenate(value,axis=0) for key,value in result.items()}


def statistical_prediction(reference,mode,n):
    distributions = propagate([reference["rho_"+mode]],pi=np.asarray(reference["pi"]))
    result = {"f":np.full(n,reference["f0"]),
              "rho_h5":np.full(n,reference["rho_"+mode]),
              "rho_h20":np.full(n,reference["rho_"+mode])}
    for h,(w,sd) in distributions.items():
        result[f"weights_h{h}"]=np.broadcast_to(w,(n,w.shape[-1])).copy()
        result[f"sigmas_h{h}"]=np.broadcast_to(reference["f0"]*sd,(n,len(sd))).copy()
    return result
