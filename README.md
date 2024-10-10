# Study_Django

## 목차 (Table of Contents)

### Django 프로젝트 구조 (Django Project Structure)

- **www1**
  - **config**: Django 설정 파일
    - `asgi.py`: ASGI 설정
    - `settings.py`: 프로젝트 설정
    - `urls.py`: URL 라우팅
    - `wsgi.py`: WSGI 설정
  - **kwlee**: 애플리케이션
    - `admin.py`: 관리자 설정
    - `apps.py`: 애플리케이션 설정
    - `migrations`: 데이터베이스 마이그레이션
      - `0001_initial.py`: 초기 마이그레이션
    - `models.py`: 데이터 모델 정의
    - `static`: 정적 파일
    - `templates`: 템플릿 파일
      - `input_page.html`: 입력 페이지
      - `insert.html`: 데이터 삽입 페이지
      - `predict.html`: 예측 결과 페이지
      - `result.html`: 결과 페이지
      - `search.html`: 검색 페이지
      - `upload.html`: 파일 업로드 페이지
    - `tests.py`: 테스트 코드
    - `urls.py`: URL 라우팅
    - `views.py`: 뷰 함수
  - **manage.py**: 프로젝트 관리 명령어 실행 파일

- **www2**
  - **config**: Django 설정 파일
  - `db.sqlite3`: SQLite 데이터베이스
  - **iris**: 애플리케이션
    - `admin.py`: 관리자 설정
    - `apps.py`: 애플리케이션 설정
    - `migrations`: 데이터베이스 마이그레이션
    - `models.py`: 데이터 모델 정의
    - `templates`: 템플릿 파일
      - `iris_input.html`: 입력 페이지
    - `tests.py`: 테스트 코드
    - `views.py`: 뷰 함수
  - `iris.csv`: 데이터 파일
  - **manage.py**: 프로젝트 관리 명령어 실행 파일

## 학습 파일 (Files)

| 파일명                     |
|----------------------------|
| `240829.py`                |
| `240902.py`                |
| `README.md`                |
| `www1/manage.py`           |
| `www1/config/settings.py`   |
| `www1/kwlee/models.py`     |
| `www2/manage.py`           |
| `www2/iris/models.py`      |
| `www2/iris/iris.csv`       |

## 설치 및 실행 방법 (Installation and Usage)

1. 저장소를 클론합니다: git clone https://github.com/MinhyeongKim0412/Study_Django.git
2. Django가 설치되어 있는지 확인합니다. 설치가 필요하다면 다음 명령어로 설치합니다: pip install django
3. 각 애플리케이션 폴더에서 마이그레이션을 적용하고 서버를 실행합니다: python manage.py migrate python manage.py runserver

