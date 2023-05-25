# config/__init__.py

import os

# 기본 설정
DEBUG = False
SECRET_KEY = os.environ.get('SECRET_KEY')
DATABASE_URI = os.environ.get('DATABASE_URI')

# 환경별 설정 로드
if os.environ.get('FLASK_ENV') == 'development':
    from .development import *
elif os.environ.get('FLASK_ENV') == 'production':
    from .production import *
