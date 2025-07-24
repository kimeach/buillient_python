# buillient_python

파이썬 백엔드 예제 프로젝트입니다. 간단한 FastAPI 서버와 퀀트 분석 도구를 포함합니다.

## 사용 방법

### FastAPI 서버 실행

```
uvicorn main:app --reload
```

### 퀀트 분석 예제 실행

CSV 파일을 인자로 전달하면 이동평균선, RSI, 볼린저 밴드를 계산합니다.

```
python sample_usage.py data.csv
```

CSV 파일에는 `close` 열이 포함되어 있어야 합니다.
