"""Independent synthetic audit of data.py and models.py (not written by me).

Reads source/provenance only. Builds artificial dates/prices/covariates in
memory and synthetic parser fixtures in a temporary directory. The only
optimizer operations are exactly two synthetic PatchTST steps.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import tempfile
import traceback
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch

import data as subject_data
import models as subject_models


ROOT = Path(__file__).resolve().parent.parent
CHECKS = []
SYNTHETIC_STEPS = 0


def check(name, condition, **detail):
    ok = bool(condition)
    CHECKS.append(dict(name=name, passed=ok, **detail))
    if not ok:
        raise AssertionError(name)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def synthetic_market(auction):
    # Irregular artificial observation dates, unrelated to any source market.
    all_dates = np.arange(np.datetime64('2017-01-01'), np.datetime64('2024-01-01'))
    kept = np.arange(len(all_dates)) % 5 != 4
    dates = all_dates[kept].astype(str)
    index = np.arange(len(dates), dtype=float)
    change = .3 * np.sin(index * .4) + .2 * np.cos(index * .11)
    change[np.arange(len(change)) % 4 == 0] = 0.0
    change[114] = 1.0
    change[115:191] = 0.0
    change[0] = 0.0
    prices = 100.0 + np.cumsum(change)
    cov = np.column_stack((1e6 + 1e3 * np.sin(index), 2.0 + .1 * np.cos(index), 20.0 + index % 7)) if auction else np.empty((len(dates), 0))
    names = ('auction_volume', 'cover_ratio', 'bidder_count') if auction else ()
    days = np.array([date.fromisoformat(d).toordinal() for d in dates])
    return subject_data.Market('SYNTHETIC_AUCTION' if auction else 'SYNTHETIC_PRICE',
                               dates, days, prices, cov, names, 'synthetic-no-market-file')


def prefix(market, last):
    return dataclasses.replace(market, dates=market.dates[:last + 1].copy(),
                               days=market.days[:last + 1].copy(), price=market.price[:last + 1].copy(),
                               covariates=market.covariates[:last + 1].copy())


def reverse_state_oracle(price, origin):
    # Backward search, independently of the implementation's forward scan.
    for j in range(origin, 0, -1):
        difference = float(price[j] - price[j - 1])
        if abs(difference) > 1e-10:
            sign = -1 if difference < 0 else 1
            return sign, min(origin - j, 2), (sign + 1) * 3 + min(origin - j, 2)
    return 0, min(origin, 2), 3 + min(origin, 2)


def audit_data():
    output = []
    for auction in (False, True):
        market = synthetic_market(auction)
        for year in (2021, 2022, 2023):
            d = subject_data.AnnualData(market, year)
            before = int(np.count_nonzero(market.dates < f'{year}-01-01'))
            through = int(np.count_nonzero(market.dates < f'{year + 1}-01-01'))
            expected_val = np.arange(before - 100, before - 20)
            expected_train = np.arange(60, expected_val[0] - 20)
            expected_test = np.arange(before, through - 20)
            label = f'{market.name}/{year}'
            check(label + '/train_support_exact', np.array_equal(d.train, expected_train))
            check(label + '/validation_support_exact', np.array_equal(d.validation, expected_val))
            check(label + '/test_support_exact', np.array_equal(d.test, expected_test))
            check(label + '/train_purged', int(d.train.max()) + 20 < int(d.validation.min()))
            check(label + '/validation_mature_before_year', int(d.validation.max()) + 20 < before)
            all_target_dates = market.dates[d.test[:, None] + np.arange(1, 21)]
            check(label + '/all_twenty_targets_in_year', np.all(np.char.startswith(all_target_dates, str(year))))
            check(label + '/statistics_end_is_last_train_label_plus1', d.stats_end == int(d.train[-1]) + 21)
            train = d.batch(d.train, available_through=int(d.validation[0]) - 1)
            check(label + '/raw_targets_manual', np.array_equal(train['delta'], market.price[d.train[:, None] + np.arange(1, 21)] - market.price[d.train, None]))
            check(label + '/units_no_sqrt_h', np.array_equal(train['y'] * train['scale'][:, None], train['delta']) or np.allclose(train['y'] * train['scale'][:, None], train['delta'], rtol=1e-15, atol=1e-15))
            reference = np.mean(train['delta'] ** 2, axis=0)
            reference_floor = max(float(np.mean(train['scale'] ** 2)) * 1e-8, 1e-16)
            check(label + '/risk_denominator_training_only', np.array_equal(subject_data.risk_denominator(train), np.maximum(reference, reference_floor)))
            output.append(d.support_receipt())

        d = subject_data.AnnualData(market, 2022)
        q = int(d.test[len(d.test) // 3])
        truncated = subject_data.AnnualData(prefix(market, q), 2022)
        poisoned = dataclasses.replace(market, dates=market.dates.copy(), days=market.days.copy(),
                                       price=market.price.copy(), covariates=market.covariates.copy())
        for j in range(q + 1, len(poisoned.price)):
            new_date = date(2099, 1, 1) + timedelta(days=j - q - 1)
            poisoned.dates[j] = new_date.isoformat()
            poisoned.days[j] = new_date.toordinal()
        poisoned.price[q + 1:] = 1e6 + np.arange(len(poisoned.price) - q - 1) % 13
        poisoned.covariates[q + 1:] = 1e9
        changed = subject_data.AnnualData(poisoned, 2022)
        original_batch = d.batch([q], available_through=q, labels=False)
        for alternative, tag in ((truncated, 'prefix'), (changed, 'poisoned_price_date_cov_suffix')):
            alt = alternative.batch([q], available_through=q, labels=False)
            for field in ('origin', 'x', 'scale', 'state', 'base'):
                check(market.name + '/' + tag + '/' + field, np.array_equal(original_batch[field], alt[field]))
        for fail_ids, available, labels in (([q], q, True), ([q + 1], q, False)):
            try:
                d.batch(fail_ids, available_through=available, labels=labels)
            except ValueError:
                check(market.name + '/unavailable_request_rejected/' + str(labels), True)
            else:
                check(market.name + '/unavailable_request_rejected/' + str(labels), False)

        # Independent manual feature reconstruction including long-prefix state.
        t = q
        scale = max(float(np.sqrt(np.mean(np.diff(market.price[t - 60:t + 1]) ** 2))), d.floor)
        state_oracles = [reverse_state_oracle(market.price, j) for j in range(t - 19, t + 1)]
        manual = np.column_stack((np.diff(market.price[t - 20:t + 1]) / scale,
                                  np.log1p(np.diff(market.days[t - 20:t + 1])) / np.log(8.0),
                                  np.full(20, np.log(scale / d.train_rms)),
                                  [a[0] for a in state_oracles], [a[1] / 2.0 for a in state_oracles]))
        if auction:
            raw_cov = np.log(market.covariates[:d.stats_end])
            standardized = (np.log(market.covariates[t - 19:t + 1]) - raw_cov.mean(0)) / np.maximum(raw_cov.std(0), 1e-6)
            manual = np.column_stack((manual, standardized))
        actual_x, actual_scale, actual_state = d.at(t)
        check(market.name + '/feature_formula_and_indices', np.array_equal(actual_x, np.clip(manual, -10, 10)))
        check(market.name + '/scale_and_state_oracle', actual_scale == scale and actual_state == state_oracles[-1][2])
        sx, _, ss = d.at(180)
        check(market.name + '/last_sign_survives_more_than20_zeros', ss == 8 and np.all(sx[:, 3] == 1.0) and np.all(sx[:, 4] == 1.0))
        const = dataclasses.replace(market, price=np.full(len(market.price), 100.0))
        constant = subject_data.AnnualData(const, 2022)
        _, _, constant_state = constant.at(q)
        check(market.name + '/constant_full_prefix_state5', constant_state == 5)

    # Minimal parser guard fixtures; these are invented synthetic CSVs only.
    with tempfile.TemporaryDirectory(prefix='smp_synthetic_parser_') as temporary:
        file = Path(temporary) / 'SYNTHETIC_ONLY.csv'
        body = b'date,price\n2023-01-01,100\n2023-01-02,101\n'
        file.write_bytes(body)
        spec = dict(name='SYNTHETIC_ONLY', path=str(file), sha256=hashlib.sha256(body).hexdigest(), with_auction=False)
        check('synthetic_parser/development_accepted', len(subject_data.load_market(spec).price) == 2)
        try:
            subject_data.load_market(dict(spec, sha256='0' * 64))
        except ValueError:
            check('synthetic_parser/wrong_hash_rejected', True)
        else:
            check('synthetic_parser/wrong_hash_rejected', False)
        body = b'date,price\n2023-12-31,100\n2024-01-01,101\n'
        file.write_bytes(body)
        spec['sha256'] = hashlib.sha256(body).hexdigest()
        try:
            subject_data.load_market(spec)
        except ValueError:
            check('synthetic_parser/post2023_suffix_rejected', True)
        else:
            check('synthetic_parser/post2023_suffix_rejected', False)
    return output


def audit_vendor():
    manifest = json.loads((ROOT / 'provenance/PATCHTST_SOURCE_MANIFEST.json').read_text())
    check('vendor/fixed_author_commit', manifest['author_commit'] == '204c21efe0b39603ad6e2ca640ef5896646ab1a9')
    check('vendor/Apache_license_and_no_benchmark_claim', manifest['license'] == 'Apache-2.0' and not manifest['adapter_is_author_benchmark'])
    # Original Git blobs obtained earlier from the pinned author repository.
    original = {
        'LICENSE': '261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64',
        'PatchTST_supervised/models/PatchTST.py': '5906049dbc663a154160c289b7a237899007ed5d',
        'PatchTST_supervised/layers/PatchTST_backbone.py': '55f79561f33aa16e40a9080347c7bbce6dfd0b52',
        'PatchTST_supervised/layers/PatchTST_layers.py': '11b5bd68a94c25cfcacf2e1eb0b469ae24342408',
        'PatchTST_supervised/layers/RevIN.py': 'f7fed8176617fd21063b30798e1d2931473c4b33',
    }
    for item in manifest['files']:
        path = ROOT / 'source' / item['path']
        body = path.read_bytes()
        actual_blob = hashlib.sha1(b'blob ' + str(len(body)).encode() + b'\0' + body).hexdigest()
        check('vendor/exact_sha256/' + item['path'], digest(path) == item['sha256'])
        check('vendor/original_git_blob/' + item['path'], actual_blob == original[item['path'].removeprefix('vendor/')])
    check('vendor/exactly_four_core_files_plus_license', len(manifest['files']) == 5)
    return manifest


def audit_models():
    global SYNTHETIC_STEPS
    records = []
    patch_parameter_counts = []
    for channels in (5, 8):
        generator = torch.Generator().manual_seed(123 + channels)
        x = torch.randn(9, 20, channels, generator=generator, dtype=torch.float64)
        original_x = x.clone()
        state = torch.arange(9)
        nets = {}
        for family in subject_models.FAMILIES:
            torch.manual_seed(17)
            net = subject_models.make_model(family, channels, q0=.42, mean_abs_nonzero=.8).double()
            net.eval()
            captured = []
            entry = net.core if family == 'patchtst_native' else net.encoder
            hook = entry.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
            with torch.no_grad():
                prediction, aux = net(x, state, return_aux=True)
            hook.remove()
            check(f'model/{channels}/{family}/exact_common_X', len(captured) == 1 and torch.equal(captured[0], original_x) and torch.equal(x, original_x))
            check(f'model/{channels}/{family}/initial_RW_and_shape', prediction.shape == (9, 20) and torch.equal(prediction, torch.zeros_like(prediction)))
            nets[family] = (net, aux)
            parameters = sum(p.numel() for p in net.parameters())
            records.append(dict(channels=channels, family=family, parameters=parameters, auxiliary_weight=subject_models.auxiliary_weight(family)))
        base_encoder = nets['smp_full'][0].encoder.state_dict()
        for family, (net, _) in nets.items():
            if family in ('lstm_native', 'patchtst_native'):
                continue
            same = all(torch.equal(value, net.encoder.state_dict()[name]) for name, value in base_encoder.items())
            check(f'model/{channels}/{family}/identical_GRU_seed_initialization', same)
        base_aux = nets['smp_full'][1]
        comparison_aux = nets['gru_aux'][1]
        for field in ('q', 'b', 'q_current', 'b_current', 'log_q_current'):
            check(f'model/{channels}/GRUaux_matches_SMP/{field}', torch.equal(base_aux[field], comparison_aux[field]))
        for family in ('smp_no_aux', 'smp_one_step'):
            state_a = nets['smp_full'][0].state_dict()
            state_b = nets[family][0].state_dict()
            check(f'model/{channels}/{family}/exact_full_state_dict', state_a.keys() == state_b.keys() and all(torch.equal(state_a[k], state_b[k]) for k in state_a))
        check(f'model/{channels}/auxiliary_controls', subject_models.auxiliary_weight('smp_full') == .05 and subject_models.auxiliary_weight('gru_aux') == .05 and subject_models.auxiliary_weight('smp_no_aux') == 0 and subject_models.auxiliary_weight('gru_native') == 0)
        patch = nets['patchtst_native'][0]
        check(f'model/{channels}/PatchTST_explicit_adapter', patch.config.revin is False and patch.config.seq_len == 20 and patch.config.pred_len == 20 and patch.readout.weight.shape == (1, channels))
        patch_parameter_counts.append((channels, sum(p.numel() for p in patch.core.parameters()), sum(p.numel() for p in patch.readout.parameters())))
    check('model/PatchTST_shared_channel_core_parameter_count', patch_parameter_counts[0][1] == patch_parameter_counts[1][1] and [p[2] for p in patch_parameter_counts] == [6, 9])

    # Exactly two purely synthetic optimizer steps, no weight decay. The first
    # zero readout blocks core parameter gradients; its update opens the path.
    torch.manual_seed(701)
    net = subject_models.make_model('patchtst_native', 8).double().train()
    x = torch.randn(7, 20, 8, dtype=torch.float64)
    target = torch.randn(7, 20, dtype=torch.float64)
    optimizer = torch.optim.SGD(net.parameters(), lr=.03, weight_decay=0.0)
    initial_core = {k: v.detach().clone() for k, v in net.core.named_parameters()}
    gradient_receipt = []
    for step in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        prediction = net(x)
        loss = ((prediction - target) ** 2).mean()
        loss.backward()
        core_grad = sum(float(p.grad.square().sum()) for p in net.core.parameters() if p.grad is not None) ** .5
        readout_grad = sum(float(p.grad.square().sum()) for p in net.readout.parameters() if p.grad is not None) ** .5
        finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in net.parameters())
        check(f'PatchTST_synthetic_step{step}/finite', finite)
        if step == 1:
            check('PatchTST_initial_zero_readout_blocks_core_gradient', core_grad == 0.0 and readout_grad > 0.0)
        else:
            check('PatchTST_readout_update_opens_core_gradient', core_grad > 1e-12)
        optimizer.step()
        SYNTHETIC_STEPS += 1
        max_change = max(float((p.detach() - initial_core[k]).abs().max()) for k, p in net.core.named_parameters())
        if step == 1:
            check('PatchTST_first_step_no_core_parameter_change_without_decay', max_change == 0.0)
        else:
            check('PatchTST_second_step_core_parameters_change', max_change > 0.0)
        gradient_receipt.append(dict(step=step, loss=float(loss.detach()), core_gradient_norm=core_grad,
                                     readout_gradient_norm=readout_grad, max_core_parameter_change=max_change))
    return records, gradient_receipt


def main():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    sources = ('data.py', 'models.py', 'smp_model.py')
    initial_hashes = {name: digest(ROOT / 'source' / name) for name in sources}
    receipt = dict(scope='Independent source and pure synthetic audit; no market input or confirmation data.',
                   market_files_read=0, confirmation_files_read=0, market_fits=0,
                   synthetic_optimizer_steps=0, sources=initial_hashes,
                   audit_script_sha256=digest(__file__), torch_version=torch.__version__, numpy_version=np.__version__)
    failure = None
    try:
        receipt['synthetic_supports'] = audit_data()
        receipt['vendor_identity'] = audit_vendor()
        receipt['models'], receipt['PatchTST_gradient_path'] = audit_models()
        check('sources_unchanged_during_audit', initial_hashes == {name: digest(ROOT / 'source' / name) for name in sources})
        check('exactly_two_synthetic_steps', SYNTHETIC_STEPS == 2)
        receipt['status'] = 'PASS'
    except Exception:
        failure = traceback.format_exc()
        receipt['status'] = 'FAIL'
        receipt['failure'] = failure
    receipt['checks'] = CHECKS
    receipt['check_count'] = len(CHECKS)
    receipt['synthetic_optimizer_steps'] = SYNTHETIC_STEPS
    receipt['limits'] = [
        'No market fitting, price source retrieval, or confirmation access; artificial CSV fixtures are not market reads.',
        'Fixed training-prefix normalization is an offline transform; suffix causality checked at validation/evaluation issuance, not falsely imposed on every historical training origin.',
        'Input uses20tokens plus a60increment causal scale and inherited full-prefix sign; it is not a claim of only20observations information.',
        'PatchTST upstream four-file core and Apache license are unmodified; C-to1 readout and RevIN=False are this study adapters, not an author benchmark reproduction.',
        'This receipt does not audit fit.py losses, early stopping, optimization schedules, or market scores.',
        'The two synthetic optimizer steps validate a gradient path only, not statistical efficacy.',
    ]
    destination = ROOT / 'audits/DATA_BASELINES_AUDIT.json'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': receipt['status'], 'checks': len(CHECKS), 'synthetic_optimizer_steps': SYNTHETIC_STEPS, 'receipt': str(destination)}))
    if failure:
        raise RuntimeError(failure)


if __name__ == '__main__':
    main()
