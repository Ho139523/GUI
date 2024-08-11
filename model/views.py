from django.shortcuts import render

def home(request):
    context={}
    return render(request, 'model/home.html', context=context)
    
    
def result(request):
    string = 'Satisfied'
    context={'condition': string}
    return render(request, 'model/result.html', context=context)