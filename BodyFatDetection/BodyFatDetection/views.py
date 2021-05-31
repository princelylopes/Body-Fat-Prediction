from django.http import HttpResponse
from django.shortcuts import render
import joblib

def  home(request):
    return render(request,"home.html")

def  predict(request):
    lr=joblib.load("Linear_body_fat_detection_model.sav")

    lis = []

    lis.append(request.POST['a'])
    lis.append(request.POST['b'])
    lis.append(request.POST['c'])
    lis.append(request.POST['d'])
    lis.append(request.POST['e'])
    
    print(lis)

    ans = lr.predict([lis])

    return render(request,"predict.html",{'ans':ans})