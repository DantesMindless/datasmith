from django.contrib import admin
from .models.main import NeuralNetworkModel


@admin.register(NeuralNetworkModel)
class NeuralNetworkModelAdmin(admin.ModelAdmin):
    list_display = ("id",)
