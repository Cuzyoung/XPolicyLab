import torch


def subtask_span_mask(
    input_ids: torch.Tensor,
    think_close_id: int | None,
    whitespace_ids: tuple[int, ...] = (),
) -> torch.Tensor:
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    if think_close_id is None:
        return mask
    positions = (input_ids == think_close_id).nonzero(as_tuple=True)[0]
    if positions.numel() == 0:
        return mask
    start = int(positions[-1]) + 1
    ws = set(whitespace_ids)
    while start < input_ids.shape[0] and int(input_ids[start]) in ws:
        start += 1
    mask[start:] = True
    return mask


def answer_span_token_ids(tokenizer) -> tuple[int | None, tuple[int, ...]]:
    close = tokenizer.convert_tokens_to_ids("</think>")
    think_close_id = close if isinstance(close, int) and close >= 0 else None
    ws: list[int] = []
    for text in ("\n", "\n\n"):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            ws.append(int(ids[0]))
    return think_close_id, tuple(dict.fromkeys(ws))
