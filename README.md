
# 폴더 구조에 대한 설명
// https://wikidocs.net/175214

    project/
    ├── app/
    │   ├── __init__.py
    │   ├── models.py
    │   ├── views.py
    │   ├── controllers.py
    │   ├── routes.py
    │   ├── utils.py
    │   └── ...
    ├── config/
    │   ├── __init__.py
    │   ├── development.py
    │   ├── production.py
    │   └── ...
    ├── migrations/
    │   ├── ...
    ├── tests/
    │   ├── ...
    ├── requirements.txt
    └── run.py

### app/: 애플리케이션의 주요 코드와 로직이 포함된 디렉토리입니다.
#### __init__.py: Flask 애플리케이션의 초기화 및 설정을 포함하는 파일입니다.
#### models.py: 데이터베이스 모델을 정의하는 파일입니다.
#### views.py 또는 controllers.py: API 엔드포인트의 비즈니스 로직을 처리하는 뷰 또는 컨트롤러 함수를 정의하는 파일입니다.
#### routes.py: API 엔드포인트의 URL 라우팅을 처리하는 파일입니다.
#### utils.py: 유틸리티 함수 또는 도우미 함수를 포함하는 파일입니다.
#### 필요에 따라 추가적인 서브모듈이나 디렉토리를 만들 수 있습니다.
### config/: 환경별 설정 파일을 포함하는 디렉토리입니다.
#### __init__.py: 설정 파일을 로드하는 초기화 파일입니다.
#### development.py, production.py 등: 각 환경에 따른 설정 파일입니다. 환경 변수, 데이터베이스 연결, 로깅 등을 설정합니다.
#### migrations/: 데이터베이스 마이그레이션 파일을 관리하는 디렉토리입니다. SQLAlchemy 등의 ORM을 사용하는 경우 유용합니다.

### tests/: 테스트 코드를 작성하는 디렉토리입니다. 단위 테스트, 통합 테스트 등을 관리합니다.

### requirements.txt: 필요한 파이썬 패키지 및 버전을 명시하는 파일입니다. pip 명령을 통해 의존성을 설치할 수 있습니다.

### run.py: Flask 애플리케이션을 실행하는 엔트리 포인트입니다.****
