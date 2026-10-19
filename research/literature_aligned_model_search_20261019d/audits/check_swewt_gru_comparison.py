"""Synthetic feature-timing, interface and no-fit checks for SWEWT-GRU adapter."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

STUDY=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(STUDY/"source"))
import run_swewt_gru_comparison as module  # noqa: E402


def synthetic_base(price,energy,volume,origins,scale):
    tokens=np.column_stack((np.r_[0.,np.diff(price)],energy))/scale
    packed=np.column_stack((tokens,volume))
    return np.stack([packed[t-63:t+1] for t in origins]).astype("float32")


def main():
    torch.set_num_threads(1)
    reference=module.DEFAULT_REFERENCE/"PROTOCOL_BEFORE_RUN.json"
    frozen=json.loads(reference.read_text())["source_sha256"]
    assert all(module.common.digest(STUDY/"source"/name) == sha for name,sha in frozen.items())
    index=np.arange(160,dtype=float)
    price=20+.03*index+np.sin(.35*index)+.2*np.sin(1.2*index)
    energy=np.column_stack((np.sin(index*.13),np.cos(index*.09),np.sin(index*.27)))
    volume=np.column_stack((np.sin(index*.07),np.ones(len(index))))
    origins=np.array([64,80,100,120])
    scale=np.array([1.7,2.,3.,4.])
    packed=synthetic_base(price,energy,volume,origins,scale)
    x=module.build_sequences(price,origins,scale[0],packed)
    assert x.shape == (4,17,9) and np.isfinite(x).all()
    assert np.array_equal(x[...,4:],packed[:,-17:,1:])
    assert np.array_equal(x[:,-1,0],np.zeros(4,dtype="float32"))
    expected_raw=np.stack([(price[t-16:t+1]-price[t])/scale[0] for t in origins])
    raw_gap=float(np.max(np.abs(x[:,:,0]-expected_raw)))
    assert raw_gap < 1e-6
    altered_price,altered_energy,altered_volume=price.copy(),energy.copy(),volume.copy()
    altered_price[101:] += 50+np.sin(index[101:])
    altered_energy[101:] += 23
    altered_volume[101:,0] -= 2
    altered_packed=synthetic_base(altered_price,altered_energy,altered_volume,origins,scale)
    altered=module.build_sequences(altered_price,origins,scale[0],altered_packed)
    causal_gap=float(np.max(np.abs(x[origins<=100]-altered[origins<=100])))
    future_changed=float(np.max(np.abs(x[origins>100]-altered[origins>100])))
    assert causal_gap == 0 and future_changed > 0
    constant=module.price_sequence_at_origin(np.full(90,20.),70,scale[0])
    zero=module.price_sequence_at_origin(np.zeros(90),70,scale[0])
    assert constant.shape == zero.shape == (17,4)
    assert np.isfinite(constant).all() and np.isfinite(zero).all()
    assert np.max(np.abs(constant)) < 1e-5 and np.max(np.abs(zero)) == 0
    model_checks={}
    xb=torch.from_numpy(x)
    changed_aux=xb.clone()
    changed_aux[:,:,4:] += 5
    for family,expected_count in [("swewt_gru_price",4308),("swewt_gru_full",4788)]:
        model=module.make_swewt_model(family,17)
        count=sum(p.numel() for p in model.parameters())
        with torch.no_grad():
            initial=model(xb)
            assert initial.shape == (4,20) and torch.count_nonzero(initial) == 0
            # Fixed synthetic nonzero readout makes channel-isolation checks
            # informative. This is no optimizer step and uses no fitted label.
            model.readout.weight.copy_(torch.linspace(-.02,.02,640).reshape(20,32))
            model.readout.bias.copy_(torch.linspace(-.01,.01,20))
            y=model(xb)
            yp=model(changed_aux)
            effect=float((y-yp).abs().max())
            assert torch.isfinite(y).all() and y.shape == (4,20)
            if family.endswith("price"):
                assert effect == 0 and torch.equal(y,model(xb[:,:,:4]))
            else:
                assert effect > 0
        assert count == expected_count
        model_checks[family]={
            "parameters":count,"gru_input_channels":model.encoder.input_size,
            "prediction_shape":list(initial.shape),"initial_rw_max_abs_prediction":float(initial.abs().max()),
            "fixed_synthetic_readout_auxiliary_change_max_abs":effect,
            "potential_data_active_parameter_count_for_h1_h5_h20":count-17*33,
        }
    # Exercise the data adapter without accessing a market or fitting a model.
    fixture={
        "price":price,"origins":origins,"scale":scale,"x":packed,
        "y":np.stack([price[origins+h]-price[origins] for h in [1,5,20]],axis=1),
        "variance":np.array([1.2,3.4,7.8]),
        "masks":{"train":np.array([1,0,0,0],bool),"val":np.array([0,1,0,0],bool),
                 "final":np.array([1,1,0,0],bool),"test":np.array([0,0,1,1],bool)},
        "manifest":{"synthetic_fixture":True},
    }
    old_prepare=module.replication.prepare_replication
    try:
        module.replication.prepare_replication=lambda market,fold:copy.deepcopy(fixture)
        adapted=module.prepare_swewt("SYNTHETIC",module.replication.FOLD)
    finally:
        module.replication.prepare_replication=old_prepare
    for key in ["origins","scale","y","variance"]:
        assert np.array_equal(adapted[key],fixture[key])
    assert all(np.array_equal(adapted["masks"][k],v) for k,v in fixture["masks"].items())
    assert np.array_equal(adapted["x"],x)
    assert module.common.prepare is module.prepare_swewt
    assert module.common.make_model is module.make_swewt_model
    assert all(module.common.digest(STUDY/"source"/name) == sha for name,sha in frozen.items())
    report={
        "status":"PASS_SYNTHETIC_NO_FIT","new_market_fits":0,"new_synthetic_fits":0,"optimizer_steps":0,
        "original_v1_v3_frozen_sources_unchanged":True,"original_source_sha256":frozen,
        "comparison_source_sha256":module.common.digest(STUDY/"source"/"run_swewt_gru_comparison.py"),
        "literature_ewt_helper_sha256":module.common.digest(STUDY/"source"/"literature_diagnostic.py"),
        "synthetic_shape":list(x.shape),"future_perturbation_past_origins_feature_max_abs_gap":causal_gap,
        "future_perturbation_later_origin_effect_max_abs":future_changed,
        "origin_centered_raw_feature_max_abs_gap":raw_gap,
        "same_last_17_energy_volume_features":True,
        "constant_window_max_centered_feature_abs":float(np.max(np.abs(constant))),
        "zero_window_max_feature_abs":float(np.max(np.abs(zero))),
        "synthetic_adapter_preserves_origins_labels_masks_scales_variance":True,
        "model_checks":model_checks,"runtime":module.dependency_receipt(),
        "scope":"Feature causality and source/interface qualification only; no market preparation, market fit, synthetic training or outcome-performance measurement. Unchanged original v3 protocol was read only for source identity.",
    }
    out=STUDY/"audits/SWEWT_GRU_SYNTHETIC_CHECK.json"
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__ == "__main__":
    main()
