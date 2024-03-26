from django.shortcuts import render
import numpy as np
from .utils import get_user_input, visualize_results
from .models import DiffractionGrating, PlaneWave
from .inputform import inputForm

def my_view(request):
    if request.method == 'POST':
        form = inputForm(request.POST)

        
        # Получаем ввод пользователя
        size, wavelength, phase, potential_type, absorption = get_user_input()
        
        # Создаем объекты DiffractionGrating и PlaneWave
        absorption = 0.2  # Absorption coefficient for the imaginary potential
        wave = PlaneWave(size=size, wavelength=wavelength, phase=phase)
        potential = DiffractionGrating(size=(size, size), absorption=absorption, y=30, gap_num=2, gap_size=20, gap_space=40, potential_type=potential_type)
        
        # Выполняем симуляцию
        results = wave.multislice(potential)
        
        # Визуализируем результаты
        visualize_results(results)
        
        # Возвращаем ответ пользователю
        return render(request, 'main/index.html', {'diffraction_gratings': potential, 'multislice_results': results, 'plane_waves': wave})
    else:
        return render(request, 'main/index.html')
