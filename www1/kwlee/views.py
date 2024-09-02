from django.shortcuts import render
from django.http import HttpResponse
from .models import Custom
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('C:/www1/kwlee/model.p','rb') as f:
    model = pickle.load(f)

def input_data(request):
    return render(request,"input_page.html")


def predict(request):
    sl = float(request.GET.get("sl"))
    sw = float(request.GET.get("sw"))
    pl = float(request.GET.get("pl"))
    pw = float(request.GET.get("pw"))

    dt = np.array([sl,sw,pl,pw]).reshape(1,4)
    result = model.predict(dt)[0]
    if result == 0:
        retval = 'setosa'
    elif result == 1:
        retval = 'versicolor'
    elif result == 2:
        retval = 'virginica'

    return render(request,"predict.html",{'retval':retval,'result':result})


def insert(request):
    return render(request,"insert.html")

def insert_action(request):
    name = request.GET.get("name")
    cnt = request.GET.get("count")
    Custom(name=name,cnt=cnt).save()
    return render(request,"result.html",{"name":name,"cnt":cnt})

def search(request):
    show_image()
    name = request.GET.get("name")
    cnt = request.GET.get("count")
    data = {"tio":"Search","name":name, "cnt":cnt}
    return render(request,"search.html",data)

def upload(request):
    return render(request,"upload.html")

def upload_file(request):
    file = request.FILES.get("file")
    file_name = file.name

    with open("C:/www1/kwlee/static/"+file_name,"wb") as f:
        f.write(file.read())
    return HttpResponse(file_name+" 이 잘 저장되었습니다")