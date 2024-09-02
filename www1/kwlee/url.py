from django.urls import path
from .views import insert, search, insert_action, input_data
from .views import predict, upload, upload_file

urlpatterns = [
    path("insert/",insert),
    path("insert/action/",insert_action),
    path("search/",search),
    path("iris/",input_data),
    path("predict/",predict),
    path("upload/",upload),
    path("upload/action/",upload_file),
]