from prefect import flow, task
import subprocess

@task
def run_sentiment_analysis_container():
    print("🚀 감성 분석 컨테이너 실행 중...")
    try:
        subprocess.run(
            ["docker", "run", "--rm", "-v", "./sentiment/input:/app/input", "sentiment-analysis"],
            check=True
        )
        print("✅ 감성 분석 완료")
    except subprocess.CalledProcessError as e:
        print("❌ 감성 분석 컨테이너 실패", e)
        raise

@flow
def full_pipeline():
    run_sentiment_analysis_container()

if __name__ == "__main__":
    full_pipeline()
