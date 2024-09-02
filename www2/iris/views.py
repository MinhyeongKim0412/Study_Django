from django.shortcuts import render
import pandas as pd
# Create your views here.iris_input
data = pd.read_csv("C:/www2/iris.csv")
def iris_input(request):
    return render(request,"iris_input.html")