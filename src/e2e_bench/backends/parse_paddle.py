"""
PaddleOCR result -> NormalizedWord mapper.

**Verified against a real installed PaddleOCR run (2026-07-16, paddleocr==3.7.0).** That
version's `.ocr()` is a deprecated shim over the new unified pipeline and returns a list of
`paddlex.inference.pipelines.ocr.result.OCRResult` (dict-like) objects, NOT the classic
`[[quad, (text, confidence)], ...]` shape older PP-OCRv3/v4 docs describe. Confirmed keys on
a real result: `rec_texts` (List[str]), `rec_scores` (List[float]), `rec_boxes`
(List[[x0,y0,x1,y1]] axis-aligned ints — simpler than the classic quad polygon).
"""
from ..types import NormalizedWord, ParseOutcome


def parse_paddle_ocr_result(paddle_result_for_page) -> ParseOutcome:
    """paddle_result_for_page: one page's result object from `PaddleOCR(...).ocr(path)`
    (already indexed, i.e. caller passes `result[0]`, not the whole `result` list) —
    dict-like with `rec_texts`/`rec_scores`/`rec_boxes` keys (paddleocr>=3.x)."""
    if paddle_result_for_page is None:
        return ParseOutcome.ok([], "")

    words = []
    try:
        texts = paddle_result_for_page["rec_texts"]
        scores = paddle_result_for_page["rec_scores"]
        boxes = paddle_result_for_page["rec_boxes"]
        for text, score, box in zip(texts, scores, boxes):
            x0, y0, x1, y1 = (int(round(float(v))) for v in box)
            words.append(NormalizedWord(text=text, bbox=[x0, y0, x1, y1], confidence=float(score)))
    except (KeyError, ValueError, TypeError, IndexError) as e:
        return ParseOutcome.failed(str(paddle_result_for_page)[:500],
                                   f"unexpected PaddleOCR result shape: {type(e).__name__}: {e}")
    return ParseOutcome.ok(words, "")
