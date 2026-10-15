import unittest
import numpy as np
import torch
from event_inputs import make_arrays
from event_model import EventNet


class ModelTests(unittest.TestCase):
    def test_raw_difference_control_is_not_hidden_behind_a_neural_encoder(self):
        torch.set_default_dtype(torch.float64)
        data = dict(source=torch.arange(28.).reshape(4, 7), neutral_source=torch.zeros(4, 7),
                    source_index=torch.tensor([0, 1, 1, 2, 3, 4]),
                    previous_index=torch.tensor([0, 0, 1, 1, 2, 3]))
        held = EventNet('raw_hold').messages(data)
        delta = EventNet('raw_event').messages(data)
        torch.testing.assert_close(delta.cumsum(0), held)

    def test_release_messages_telescope_without_repeating_a_held_report(self):
        torch.set_default_dtype(torch.float64)
        dates = np.arange(260) + 17000
        sd = dates[8::15]
        sv = 450 + np.arange(len(sd))[:, None] * np.array([[2., 3., 4.]])
        arrays, _ = make_arrays(30 + np.sin(np.arange(260)), dates, sd, sv, 220, 130, .2)
        data = {k: torch.as_tensor(v) for k, v in arrays.items()}
        event = EventNet('event')
        hold = EventNet('hold')
        hold.load_state_dict(event.state_dict())
        messages = event.messages(data)
        torch.testing.assert_close(messages.sum(0), hold.messages(data)[-1])
        self.assertEqual(float(messages[data['clock'][:, 0] == 0].abs().max()), 0.)
        messages.square().sum().backward()
        self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in event.source_encoder.parameters()))
        data['source'] = data['neutral_source']
        self.assertEqual(float(event.messages(data).abs().max()), 0.)


if __name__ == '__main__':
    unittest.main()
