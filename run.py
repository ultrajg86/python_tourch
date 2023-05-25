# run.py

from app import create_app

# Flask 애플리케이션 생성
app = create_app()

if __name__ == '__main__':
    # 개발 환경에서만 디버그 모드로 실행
    app.run(debug=True)
