# 가드레일(Guardrail) — AI 브랜드 검수 린터

광고 배너·카피가 브랜드 가이드와 광고 규제를 지키는지 브라우저 안에서 검수하는 정적 웹앱입니다.
2026 K-AWARDS DIGITAL EXCELLENCE AWARDS 대학부(AI 서비스 혁신) 출품작.

- 실행: 이 폴더를 정적 서버로 열면 됩니다(`python3 -m http.server`). 서버로 이미지가 전송되지 않습니다.
- AI 모델: Tesseract.js(OCR), Transformers.js + multilingual-e5-small(문장 임베딩), CLIP ViT-B/32(시각 무드)
- 예시 브랜드 ‘숲결’·‘모아핀’과 샘플 배너는 시연용 가상 자료입니다. 규칙은 예시이며 법률 자문이 아닙니다.
- 톤 판정 검증: `eval/` 페이지에서 새 문장 60개 판정을 재현할 수 있습니다.
