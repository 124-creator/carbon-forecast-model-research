#!/usr/bin/env python3
"""Independent synthetic audit of the SMP model; no market or training reads.

The reference calculations use explicit event-path enumeration and NumPy
augmented transition matrices. No reference calls the model's propagator.
Autograd is checked against finite differences; no optimizer is instantiated.
The data/runner maturity and native raw-risk loss remain a separate audit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


EXPECTED_MODEL_SHA256 = "c6f8bbd75a0a4ae025eabd1e31d6bdab1170569c154b9cc16b464172b01512c5"
VARIANTS = ("full", "no_clock", "no_sign", "pooled_mark", "iid_state", "one_step")
GROUP_COUNTS = {"full": (9,9), "no_clock": (3,3), "no_sign": (3,3),
                "pooled_mark": (9,1), "iid_state": (1,1), "one_step": (9,9)}


def next_state(state, event):
    """Independent arithmetic state definition, not imported NEXT_ZERO."""
    sign = state // 3 - 1
    age = state % 3
    if event == 0:
        return (sign + 1)*3 + min(age+1,2)
    return 0 if event == 1 else 6


def matrices(q, b):
    transition = np.zeros((9,9), dtype=np.float64)
    reward = np.zeros(9, dtype=np.float64)
    for state in range(9):
        for event, amount in enumerate((0., -b[state,0], b[state,1])):
            transition[state, next_state(state,event)] += q[state,event]
            reward[state] += q[state,event]*amount
    return transition, reward


def matrix_oracle(q, b, state, horizon=20):
    """The upper-right block of [[P,r],[0,1]]**h is sum P**k r."""
    transition, reward = matrices(q,b)
    augmented = np.zeros((10,10), dtype=np.float64)
    augmented[:9,:9] = transition
    augmented[:9,9] = reward
    augmented[9,9] = 1.
    return np.array([np.linalg.matrix_power(augmented,h)[state,9] for h in range(1,horizon+1)])


def enumerate_oracle(q, b, state, horizon):
    total = 0.
    for events in itertools.product(range(3), repeat=horizon):
        probability, amount, current = 1., 0., state
        for event in events:
            probability *= q[current,event]
            amount += (0., -b[current,0], b[current,1])[event]
            current = next_state(current,event)
        total += probability*amount
    return total


def stable_softmax(x):
    shifted=x-x.max(axis=-1,keepdims=True)
    ex=np.exp(shifted)
    return ex/ex.sum(axis=-1,keepdims=True)


def full_prefix_state(changes, epsilon=1e-10):
    """Synthetic oracle only. Production preprocessing is not imported here."""
    sign, age = 0, 0
    for change in changes:
        if abs(change) > epsilon:
            sign, age = (-1 if change < 0 else 1), 0
        else:
            age = min(age+1,2)
    return (sign+1)*3+age


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,default=Path(__file__).with_name("smp_model.py"))
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[1]/"audits/SMP_MATH_AUDIT.json")
    args=parser.parse_args()
    model_bytes=args.model.read_bytes()
    model_hash=hashlib.sha256(model_bytes).hexdigest()
    if model_hash != EXPECTED_MODEL_SHA256:
        raise ValueError("Model source hash changed; review the new bytes before updating the audit pin")
    spec=importlib.util.spec_from_file_location("independently_audited_smp",args.model)
    model_module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.set_default_dtype(torch.float64)
    rng=np.random.default_rng(20261017)
    checks=[]
    comparison_count=0
    maxima={}

    def check(name,condition,details=None):
        if not bool(condition):
            raise AssertionError(f"{name}: {details}")
        checks.append({"name":name,"status":"PASS"})

    def close(name,actual,expected,atol=1e-10,rtol=1e-9,category="numeric"):
        nonlocal comparison_count
        a=np.asarray(actual,dtype=np.float64); e=np.asarray(expected,dtype=np.float64)
        check(name+" shape",a.shape==e.shape,{"actual":a.shape,"expected":e.shape})
        error=float(np.max(np.abs(a-e))) if a.size else 0.
        maxima[category]=max(maxima.get(category,0.),error)
        comparison_count+=a.size
        check(name+" values",np.allclose(a,e,atol=atol,rtol=rtol),{"max_abs_error":error})

    def raises(name,fn,exception=(ValueError,TypeError)):
        try:
            fn()
        except exception:
            check(name,True)
        else:
            check(name,False)

    # Arbitrary positive probability/mark tables; every general API initial state.
    logits=rng.normal(size=(9,9,3))*.6
    marks=rng.normal(size=(9,9,2))*.3
    q=stable_softmax(logits)
    b=np.logaddexp(0.,marks)+model_module.MIN_MARK_SCALE
    states=np.arange(9,dtype=np.int64)
    predicted=model_module.propagate_mean(torch.tensor(q),torch.tensor(b),torch.tensor(states),20).detach().numpy()
    oracle=np.stack([matrix_oracle(q[i],b[i],int(states[i])) for i in range(9)])
    close("generic H20 propagator vs independent augmented matrix",predicted,oracle,category="matrix_forward")
    for i in range(9):
        for h in range(1,6):
            expected=enumerate_oracle(q[i],b[i],i,h)
            close(f"general state{i} H{h} exhaustive event-path mean",predicted[i,h-1],expected,category="enumeration_forward")
        transition,_=matrices(q[i],b[i])
        close(f"general case{i} transition row sums",transition.sum(1),np.ones(9),atol=1e-14,category="stochastic_kernel")

    # Gradients of the mathematical propagator, finite-differenced through an
    # independent NumPy oracle, including reachability-induced zero derivatives.
    lt=torch.tensor(logits,requires_grad=True)
    bt=torch.tensor(marks,requires_grad=True)
    out=model_module.propagate_mean(torch.softmax(lt,dim=-1),torch.nn.functional.softplus(bt)+model_module.MIN_MARK_SCALE,torch.tensor(states),20)
    objective_weights=np.linspace(.1,.7,20)[None,:]
    objective=(out*torch.tensor(objective_weights)).mean()+.03*out.square().mean()
    gradients=torch.autograd.grad(objective,(lt,bt))

    def oracle_objective(logit_arg,mark_arg):
        probs=stable_softmax(logit_arg)
        mags=np.logaddexp(0.,mark_arg)+model_module.MIN_MARK_SCALE
        m=np.stack([matrix_oracle(probs[i],mags[i],i) for i in range(9)])
        return float((m*objective_weights).mean()+.03*(m*m).mean())

    epsilon=1e-6
    prop_fd=[]
    for name,array,gradient in [("logits",logits,gradients[0].detach().numpy()),("marks",marks,gradients[1].detach().numpy())]:
        ranked=np.argsort(np.abs(gradient).ravel())
        picked=set(int(i) for i in ranked[-6:])
        picked.update(int(i) for i in ranked[:2])
        for flat in sorted(picked):
            idx=np.unravel_index(flat,array.shape)
            plus=array.copy();minus=array.copy();plus[idx]+=epsilon;minus[idx]-=epsilon
            if name=="logits":
                numeric=(oracle_objective(plus,marks)-oracle_objective(minus,marks))/(2*epsilon)
            else:
                numeric=(oracle_objective(logits,plus)-oracle_objective(logits,minus))/(2*epsilon)
            close(f"independent matrix finite difference {name}{idx}",gradient[idx],numeric,atol=2e-8,rtol=2e-5,category="independent_propagation_gradient")
            prop_fd.append({"parameter":name,"index":[int(i) for i in idx],"autograd":float(gradient[idx]),"finite_difference":float(numeric)})

    # Actual network variants. All source-state outputs are inspected after a
    # head perturbation; checking only symmetric initialization would be vacuous.
    variants=[]
    channels,hidden=7,32
    inputs=torch.tensor(rng.normal(size=(9,20,channels))*.25)
    network_states=torch.arange(9,dtype=torch.long)
    reference_full=None
    network_fd=[]
    for variant in VARIANTS:
        torch.manual_seed(113)
        net=model_module.SMPModel(channels,hidden,20,variant).double()
        q_groups,b_groups=GROUP_COUNTS[variant]
        check(variant+" declared free group counts",(net.n_q_groups,net.n_b_groups)==(q_groups,b_groups))
        expected_encoder=3*hidden*(channels+hidden+2)
        expected_head=(3*q_groups+2*b_groups)*(hidden+1)
        count_encoder=sum(p.numel() for p in net.encoder.parameters())
        count_head=sum(p.numel() for p in net.q_head.parameters())+sum(p.numel() for p in net.b_head.parameters())
        check(variant+" actual allocated encoder count",count_encoder==expected_encoder)
        check(variant+" actual allocated head count",count_head==expected_head)
        init_pred,init_aux=net(inputs,network_states,True)
        check(variant+" exact RW initial forecast",torch.equal(init_pred,torch.zeros_like(init_pred)))
        close(variant+" initial positive mark means",init_aux['b'].detach().numpy(),np.ones((9,9,2)),atol=2e-15,rtol=0.,category="initialization")
        init_loss=(init_pred-torch.ones_like(init_pred)).square().mean()
        init_loss.backward()
        check(variant+" initial encoder data gradient is exactly zero",all(p.grad is not None and torch.count_nonzero(p.grad)==0 for p in net.encoder.parameters()))
        check(variant+" initial head data gradient is nonzero",any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in list(net.q_head.parameters())+list(net.b_head.parameters())))
        net.zero_grad(set_to_none=True)
        with torch.no_grad():
            for head in [net.q_head,net.b_head]:
                head.weight.add_(torch.tensor(rng.normal(size=tuple(head.weight.shape))*.08))
                head.bias.add_(torch.tensor(rng.normal(size=tuple(head.bias.shape))*.15))
        if variant=="full":
            reference_full={k:v.detach().clone() for k,v in net.state_dict().items()}
        if variant=="one_step":
            net.load_state_dict(reference_full)
        m,aux=net(inputs,network_states,True)
        qq=aux['q'].detach().numpy();bb=aux['b'].detach().numpy()
        close(variant+" actual network output vs matrix oracle",m.detach().numpy(),np.stack([matrix_oracle(qq[i],bb[i],i) for i in range(9)]),category="network_forward")
        q_unique=(lambda a:a//3) if variant=="no_clock" else (lambda a:a%3) if variant=="no_sign" else (lambda a:0) if variant=="iid_state" else (lambda a:a)
        b_unique=(lambda a:0) if variant in ["iid_state","pooled_mark"] else q_unique
        for a in range(9):
            for c in range(a+1,9):
                if q_unique(a)==q_unique(c):
                    check(f"{variant} actual q tie {a},{c}",np.array_equal(qq[:,a],qq[:,c]))
                if b_unique(a)==b_unique(c):
                    check(f"{variant} actual b tie {a},{c}",np.array_equal(bb[:,a],bb[:,c]))
        check(variant+" q groups differ after perturbation",q_groups==1 or np.max(np.abs(qq[:,0]-qq[:,-1]))>1e-8)
        check(variant+" b groups differ after perturbation",b_groups==1 or np.max(np.abs(bb[:,0]-bb[:,-1]))>1e-8)
        fixed_x=inputs[:1].repeat(9,1,1)
        fixed_m=net(fixed_x,network_states).detach().numpy()
        if variant=="no_clock":
            for group in range(3):
                close(f"no_clock same context sign{group-1} full forecast age invariance",fixed_m[group*3:group*3+3],np.repeat(fixed_m[group*3:group*3+1],3,axis=0),category="ablation_behavior")
        elif variant=="no_sign":
            for age in range(3):
                close(f"no_sign same context age{age} full forecast sign invariance",fixed_m[age::3],np.repeat(fixed_m[age:age+1],3,axis=0),category="ablation_behavior")
        elif variant=="iid_state":
            close("iid_state all initial states equivalent",fixed_m,np.repeat(fixed_m[:1],9,axis=0),category="ablation_behavior")
            close("iid_state exact H times single-step expected reward",m.detach().numpy(),m[:,0:1].detach().numpy()*np.arange(1,21)[None,:],category="ablation_behavior")
        elif variant=="one_step":
            full=model_module.SMPModel(channels,hidden,20,'full').double();full.load_state_dict(reference_full)
            check("one_step and full exact forward identity at same parameters",torch.equal(full(inputs,network_states),m))
        loss=(m*torch.tensor(objective_weights)).mean()+.03*m.square().mean()
        loss.backward()
        module_norms={name:float(torch.sqrt(sum((p.grad.square().sum() for p in module.parameters()),torch.tensor(0.)))) for name,module in [('encoder',net.encoder),('q_head',net.q_head),('b_head',net.b_head)]}
        check(variant+" every module has nonzero finite post-perturb gradient",all(math.isfinite(v) and v>0 for v in module_norms.values()))
        free_row_coverage={}
        for name,head in [('q_head',net.q_head),('b_head',net.b_head)]:
            row_grad=head.weight.grad.abs().sum(1)+head.bias.grad.abs()
            free_row_coverage[name]=int((row_grad>1e-12).sum())
            check(variant+" all nominal head output rows influence general-state synthetic task "+name,bool((row_grad>1e-12).all()))
        # One maximum-magnitude coordinate from each of the three modules.
        for prefix in ['encoder.','q_head.','b_head.']:
            choices=[(name,p) for name,p in net.named_parameters() if name.startswith(prefix)]
            pname,param=max(choices,key=lambda pair:float(pair[1].grad.abs().max()))
            flat=int(param.grad.abs().argmax());idx=np.unravel_index(flat,tuple(param.shape))
            analytic=float(param.grad[idx]);original=float(param.detach()[idx])

            def evaluate_at(value):
                with torch.no_grad():
                    param[idx]=value
                    pred=net(inputs,network_states)
                    return float((pred*torch.tensor(objective_weights)).mean()+.03*pred.square().mean())

            numeric=(evaluate_at(original+epsilon)-evaluate_at(original-epsilon))/(2*epsilon)
            with torch.no_grad():param[idx]=original
            close(f"{variant} network finite difference {pname}{idx}",analytic,numeric,atol=3e-8,rtol=3e-5,category="network_gradient")
            network_fd.append({"variant":variant,"parameter":pname,"index":[int(i) for i in idx],"autograd":analytic,"finite_difference":numeric})
        # The constant normalized prefix has a legal initial state5; it need not
        # remain RW after nonzero learned biases. Require finite differentiability.
        constant=torch.zeros((1,20,channels))
        cp=net(constant,torch.tensor([5]))
        check(variant+" constant prefix state5 finite after perturbation",bool(torch.isfinite(cp).all()))
        net.zero_grad(set_to_none=True);cp.square().mean().backward()
        check(variant+" constant prefix gradients finite",all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in net.parameters()))
        variants.append({'variant':variant,'synthetic_input_channels':channels,'hidden_size':hidden,
                         'q_free_groups':q_groups,'b_free_groups':b_groups,'encoder_parameters':count_encoder,
                         'head_parameters':count_head,'total_parameters':count_encoder+count_head,
                         'post_perturb_module_gradient_norms':module_norms,'synthetic_nominal_head_row_coverage':free_row_coverage})

    # Independent hurdle NLL values, explicit event mask, stable extreme logits,
    # and gradient isolation for zero vs signed observations.
    qq=torch.tensor([[.5,.2,.3]]*4,requires_grad=True)
    bb=torch.tensor([[2.,4.]]*4,requires_grad=True)
    delta=torch.tensor([0.,-1.,3.,1e-12])
    events=torch.tensor([False,True,True,False])
    nll=model_module.hurdle_nll({'q_current':qq,'b_current':bb},delta,event=events)
    expected=np.array([-math.log(.5),-math.log(.2)+math.log(2)+.5,-math.log(.3)+math.log(4)+.75,-math.log(.5)])
    close("hurdle NLL manual values and explicit tiny-event exclusion",nll.detach().numpy(),expected,category="nll")
    close("hurdle NLL mean reduction",float(model_module.hurdle_nll({'q_current':qq,'b_current':bb},delta,reduction='mean',event=events).detach()),expected.mean(),category="nll")
    close("hurdle NLL sum reduction",float(model_module.hurdle_nll({'q_current':qq,'b_current':bb},delta,reduction='sum',event=events).detach()),expected.sum(),category="nll")
    nll.sum().backward()
    expected_q_grad=np.zeros((4,3));expected_q_grad[[0,3],0]=-2.;expected_q_grad[1,1]=-5.;expected_q_grad[2,2]=-1/.3
    expected_b_grad=np.zeros((4,2));expected_b_grad[1,0]=1/2-1/4;expected_b_grad[2,1]=1/4-3/16
    close("NLL q derivative independent formula",qq.grad.numpy(),expected_q_grad,category="nll_gradient")
    close("NLL b derivative independent formula",bb.grad.numpy(),expected_b_grad,category="nll_gradient")
    extreme=torch.tensor([[700.,-900.,-800.],[-800.,700.,-900.]],requires_grad=True)
    lq=torch.log_softmax(extreme,dim=1)
    eb=torch.tensor([[.2,.3],[.2,.3]],requires_grad=True)
    extreme_nll=model_module.hurdle_nll({'q_current':lq.exp(),'b_current':eb,'log_q_current':lq},torch.tensor([-1.,1.]),event=torch.tensor([True,True]))
    expected_extreme=np.array([1600.+math.log(.2)+5.,1600.+math.log(.3)+1/.3])
    close("stable NLL retains log-softmax despite probability underflow",extreme_nll.detach().numpy(),expected_extreme,category="nll")
    extreme_nll.sum().backward()
    check("extreme NLL gradients finite",bool(torch.isfinite(extreme.grad).all() and torch.isfinite(eb.grad).all()))

    # Initialization clipping/floor explicitly reported; GRU untouched.
    init_net=model_module.SMPModel(1,hidden,20,'full').double()
    encoder_before={k:v.clone() for k,v in init_net.encoder.state_dict().items()}
    init_record=init_net.initialize_head(q0=1.,mean_abs_nonzero=0.)
    check("initialization clips q0 explicitly",init_record['applied_q0']==.95)
    check("initialization floors empty/nonpositive mark statistic explicitly",init_record['applied_mean_abs_nonzero']>model_module.MIN_MARK_SCALE)
    check("head initialization leaves GRU state unchanged",all(torch.equal(v,init_net.encoder.state_dict()[k]) for k,v in encoder_before.items()))
    all_zero=init_net(torch.zeros(9,20,1),torch.arange(9))
    check("floor/clip initialization still gives exact RW for all states",torch.equal(all_zero,torch.zeros_like(all_zero)))

    # Structural validation only, no invented probability validation contract.
    valid_q=torch.tensor(q);valid_b=torch.tensor(b);valid_state=torch.arange(9)
    raises("reject floating state",lambda:model_module.propagate_mean(valid_q,valid_b,valid_state.double()))
    raises("reject boolean state",lambda:model_module.propagate_mean(valid_q,valid_b,torch.zeros(9,dtype=torch.bool)))
    raises("reject state9",lambda:model_module.propagate_mean(valid_q,valid_b,torch.full((9,),9)))
    raises("reject negative state",lambda:model_module.propagate_mean(valid_q,valid_b,torch.full((9,),-1)))
    raises("reject wrong q shape",lambda:model_module.propagate_mean(valid_q[:,:8],valid_b,valid_state))
    raises("reject wrong b batch",lambda:model_module.propagate_mean(valid_q,valid_b[:8],valid_state))
    raises("reject fractional horizon",lambda:model_module.propagate_mean(valid_q,valid_b,valid_state,1.5))
    raises("reject horizon0",lambda:model_module.propagate_mean(valid_q,valid_b,valid_state,0))
    raises("reject bool horizon",lambda:model_module.propagate_mean(valid_q,valid_b,valid_state,True))
    raises("reject a true event with zero sign",lambda:model_module.hurdle_nll({'q_current':qq,'b_current':bb},torch.zeros(4),event=torch.ones(4,dtype=torch.bool)))
    raises("reject mismatched event mask",lambda:model_module.hurdle_nll({'q_current':qq,'b_current':bb},delta,event=torch.ones(3,dtype=torch.bool)))
    raises("reject unknown variant",lambda:model_module.SMPModel(1,variant='not_registered'))

    # Full-prefix state semantics and a suffix perturbation sanity check. These
    # prove the independent definition, not a production data pipeline yet.
    check("empty general prefix state3",full_prefix_state([])==3)
    check("single-zero general prefix state4",full_prefix_state([0.])==4)
    check("long constant prefix state5",full_prefix_state([0.]*60)==5)
    check("long zero run retains previously observed negative sign",full_prefix_state([-1.]+[0.]*60)==2)
    check("long zero run retains previously observed positive sign",full_prefix_state([1.]+[0.]*60)==8)
    check("negative event resets to0",full_prefix_state([0.,1.,0.,-1.])==0)
    check("positive event resets to6",full_prefix_state([-1.,0.,1.])==6)
    fixture=[0.,-1.,0.,0.,0.]+[1.]*20
    query=4
    original_state=full_prefix_state(fixture[:query+1])
    changed=fixture[:query+1]+[-1000.]*20
    check("prefix-state oracle invariant to unseen suffix",original_state==full_prefix_state(changed[:query+1]))
    reachable={0,1,2,5,6,7,8}
    check("market-length-prefix states form closed transition support",all(next_state(a,e) in reachable for a in reachable for e in range(3)))

    result={
        'created_utc':datetime.now(timezone.utc).isoformat(),'status':'PASS',
        'model_path':str(args.model),'model_sha256':model_hash,
        'audit_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'torch_default_dtype':str(torch.get_default_dtype()),'threads':torch.get_num_threads()},
        'check_count':len(checks),'numerical_values_compared':int(comparison_count),'maximum_absolute_errors':maxima,
        'variants':variants,'propagation_finite_difference_checks':prop_fd,'network_finite_difference_checks':network_fd,
        'check_records':checks,'initialization_floor_clip_example':init_record,
        'activity':{'market_data_reads':0,'confirmation_reads':0,'weight_or_prediction_file_reads':0,'market_fits':0,'optimizer_instances':0,'optimizer_steps':0,'old_file_writes':0},
        'scope_limits':[
            'Synthetic fixtures include all nine general API states. Market-length prefixes normally reach only states 0,1,2,5,6,7,8; nominal rows3/4 are not evidence of actual market capacity.',
            'no_clock/no_sign tie future source-state groups at fixed current context. They do not remove age/sign information from the current x/context.',
            'one_step is exactly full forward architecture. The runner must independently establish that only its h1 objective is used.',
            'The source computes an exact expected reward in its specified finite state model with current GRU context fixed. This is standard Markov reward mathematics, not a novelty certificate.',
            'The independent prefix-state oracle validates the required definition. Actual feature extraction, source availability, label maturity, sample pairing, and raw-price risk weighting require the later runner/data audit.',
            'The model accepts a caller-supplied input tensor and state; there is no market or future label access in the model module. It cannot certify that a caller constructed those inputs causally.',
            'Direct propagate_mean callers own q/b validity; the source does not silently renormalize arbitrary supplied probabilities.',
            'No market predictive performance or statistical superiority was assessed.'
        ]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'status':result['status'],'checks':len(checks),'values':int(comparison_count),'maximum_absolute_errors':maxima,'model_sha256':model_hash,'activity':result['activity']},ensure_ascii=False))


if __name__=='__main__':
    main()
