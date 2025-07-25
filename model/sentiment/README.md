# Docker 이미지 빌드
cd sentiment
docker build -t sentiment-analysis .

# Docker 실행
docker run --rm -v $(pwd)/sentiment/input:/app/input sentiment-analysis
