"""Pure SSE parsing: no HTTP client, network or credentials."""
import unittest

from lari.hermes.events import (
    HermesApprovalRequest, HermesReply, HermesSSEState, HermesStatus,
    HermesTextDelta, HermesTurnCompleted, parse_hermes_lines,
)


STOP = 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}'
DONE = 'data: [DONE]'


class HermesEventsTests(unittest.TestCase):
    def test_partial_chunks_preserve_state_without_mutating_inputs(self):
        initial = HermesSSEState()
        first, events = parse_hermes_lines([
            ': keepalive', '',
            'data:{"choices":[{"delta":{"content":"Ciao"}}]}',
        ], initial, final=False)
        self.assertEqual(events, [])
        self.assertEqual(initial, HermesSSEState())
        second, events = parse_hermes_lines([
            '', 'data: {"choices":[{"delta":{"content":" mondo "}}]}', '',
        ], first, final=False)
        self.assertEqual(events, [HermesTextDelta('Ciao'), HermesTextDelta(' mondo ')])
        self.assertEqual(first.content, ())
        _, events = parse_hermes_lines([STOP, '', DONE, ''], second)
        self.assertEqual(events, [HermesTurnCompleted('Ciao mondo')])

    def test_stop_and_done_complete_with_nested_run_id(self):
        _, events = parse_hermes_lines([
            'data: {"meta":[{"run_id":"run-42"}],"choices":'
            '[{"delta":{"content":" ciao "},"finish_reason":"STOP"}]}', '', DONE, '',
        ])
        self.assertEqual(events, [HermesTextDelta(' ciao '), HermesTurnCompleted('ciao', 'run-42')])

    def test_missing_done_fails_even_after_stop(self):
        with self.assertRaisesRegex(RuntimeError, r'stream Hermes incompleto: manca \[DONE\]'):
            parse_hermes_lines([STOP, ''])

    def test_done_without_stop_fails(self):
        with self.assertRaisesRegex(RuntimeError, "manca finish_reason='stop'"):
            parse_hermes_lines([DONE, ''])

    def test_invalid_finish_reasons_fail_before_emitting_content(self):
        for reason, message in [('length', "finish_reason='length'"), ('error', 'terminato con errore')]:
            with self.subTest(reason=reason), self.assertRaisesRegex(RuntimeError, message):
                parse_hermes_lines([
                    'data: {"choices":[{"delta":{"content":"partial"},'
                    '"finish_reason":"' + reason + '"}]}', '', DONE, '',
                ], final=False)

    def test_incomplete_final_frame_is_parsed_before_terminal_validation(self):
        state, events = parse_hermes_lines([
            'data: {"choices":[{"delta":{"content":"partial"}}]}',
        ], final=False)
        self.assertEqual(events, [])
        state, events = parse_hermes_lines([''], state, final=False)
        self.assertEqual(events, [HermesTextDelta('partial')])
        with self.assertRaisesRegex(RuntimeError, r'manca \[DONE\]'):
            parse_hermes_lines((), state)

    def test_approval_and_status_metadata_never_become_speech(self):
        _, events = parse_hermes_lines([
            'event:hermes.tool.progress', 'data:{"run_id":"tool-run","message":"tool output"}', '',
            'event: hermes.status', 'data: {"message":"busy"}', '',
            'event: approval.request', 'data: {"run_id":"approval-run","action":"send"}', '',
            STOP, '', DONE, '',
        ])
        self.assertEqual(events, [
            HermesStatus('hermes.tool.progress', {'run_id': 'tool-run', 'message': 'tool output'}),
            HermesStatus('hermes.status', {'message': 'busy'}),
            HermesApprovalRequest({'run_id': 'approval-run', 'action': 'send'}),
            HermesTurnCompleted('', 'approval-run'),
        ])

    def test_corrupt_json_keeps_explicit_message_and_cause(self):
        with self.assertRaisesRegex(RuntimeError, 'stream Hermes non valido: JSON SSE corrotto') as raised:
            parse_hermes_lines(['data: {broken}', ''], final=False)
        self.assertIsNotNone(raised.exception.__cause__)

    def test_hermes_error_finish_keeps_existing_error_message(self):
        with self.assertRaisesRegex(RuntimeError, '^Hermes stream terminato con errore$'):
            parse_hermes_lines([
                'event: hermes.error',
                'data: {"choices":[{"finish_reason":"error"}]}', '',
            ], final=False)

    def test_multiline_data_comments_and_ignored_reasoning(self):
        _, events = parse_hermes_lines([
            'unknown: ignored', ': heartbeat',
            'data: {"choices":', ': another heartbeat',
            'data: [{"delta":{"reasoning_content":"hidden","tool_calls":[]}}]}', '',
            STOP, '', DONE, '',
        ])
        self.assertEqual(events, [HermesTurnCompleted('')])

    def test_final_done_without_blank_line_keeps_existing_behavior(self):
        _, events = parse_hermes_lines([STOP, '', DONE])
        self.assertEqual(events, [HermesTurnCompleted('')])

    def test_json_truncated_at_eof_is_corrupt(self):
        with self.assertRaisesRegex(RuntimeError, 'JSON SSE corrotto'):
            parse_hermes_lines(['data: {"choices":'])

    def test_reply_remains_string_compatible_with_both_ids(self):
        reply = HermesReply('ciao', 'session-1', 'run-1')
        self.assertEqual(reply, 'ciao')
        self.assertEqual((reply.session_id, reply.run_id), ('session-1', 'run-1'))


if __name__ == '__main__':
    unittest.main()
