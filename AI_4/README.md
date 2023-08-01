# Лабораторная работа №4 
## Влияние гиперпараметров на обучение нейронной сети
___
### Подробный отчет в соответствии с заданием [тут](/AI_4/Report4.pdf)
### [Интерфейс](/AI_4/Lab4_Part1.ipynb) настройки нейросети для распознавания мат функций
### [Интерфейс](/AI_4/Lab4_Part2.ipynb) настройки нейросети для распознавания изображений
### Задание: 
1) By changing these hyperparameters try to reach max accuracy value (at least 0.95) for Part2 model with fixed epoch count 20
2) Change 1st hyperparameter’s value from min to max with minimal step depends on your variant 
3) Show impact on result using graphs
4) Describe impact of each hyperparameter on accuracy.
5) Set hyperparameter value back to one which produced max accuracy
6) Repeat 2-5 steps for second hyperparameter  

|  Var  |  Part1 func  |  Part2 data |  Hyperparameters |  
| ----- | ------------ | ----------- | ---------------- |  
| 1 | Absolute(Sin(x)) X: -6,3..6.3 Y: 0..1.2 | MNIST | Layers count, neurons count per layer |
