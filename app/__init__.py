# app/__init__.py

from flask import Flask

def create_app():
    app = Flask(__name__)

    # 환경별 설정 로드 및 등록
    app.config.from_object('config.development')

    # 확장(extension) 초기화 및 등록
    from app.extensions import db
    db.init_app(app)

    # 라우트 등록
    from app.routes import user_routes
    app.register_blueprint(user_routes)

    return app
