from django.shortcuts import render

from core.base import build_context

def home(request):
    return render(request, 'core/home.html', build_context(request.GET))
