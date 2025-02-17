# Решения задач курса "Суперкомпьютеры и Параллельная Обработка Данных"

1) Разработка параллельной версии программы для умножения матрицы на вектор с
использованием библиотеки OpenMP

Запуск:

	$ ./omp_run.sh

2) Разработка параллельной версии программы для умножения матрицы на вектор с    
использованием библиотеки MPI.

Запуск:

	$ SIZE=1 NP=1 ./mpi_run.sh

# Решения задач курса "Распределённые системы"

3) Реализовать программу, моделирующую выполнение операции MPI_Scan для
транспьютерной матрицы размером 4`*`4 при помощи пересылок MPI типа точка-точка.
В каждом узле транспьютерной матрицы запущен один MPI-процесс. Каждый процесс
имеет свое число. В результате выполнения операции MPI_Scan каждый i-ый процесс
должен получить сумму чисел, которые находятся у процессов с номерами 0, ... i
включительно. Оценить сколько времени потребуется для выполнения операции
MPI_Scan, если все процессы выдали эту операцию редукции одновременно. Время
старта равно 100, время передачи байта равно 1 (Ts=100,Tb=1). Процессорные
операции, включая чтение из памяти и запись в память, считаются бесконечно быстрыми.

Запуск:

	$ ./scan_run.sh

4) Доработать MPI-программу, реализованную в рамках курса “Суперкомпьютеры и
параллельная обработка данных”. Добавить контрольные точки для продолжения
работы программы в случае сбоя. Реализовать один из 3-х сценариев работы после
сбоя: a) продолжить работу программы только на “исправных” процессах; б) вместо
процессов, вышедших из строя, создать новые MPI-процессы, которые необходимо
использовать для продолжения расчетов; в) при запуске программы на счет сразу
запустить некоторое дополнительное количество MPI-процессов, которые
использовать в случае сбоя.

Запуск:

	$ SIZE=2 NP=2 ./mpi_run.sh

# Решения задач курса "Параллельные (высокопроизводительные) вычисления"

5) Сгенерировать портрет разреженной матрицы на основе сетки (вариант Б2),
посторить СЛАУ по полученному портрету и решить алгоритмом предобусловленного
метода CG, распараллелить средствами openMP.

Сборка:

	$ make

Если программа не собирается, можно использовать следующий патч, заменяющий
`clang++` на `g++` и `c++17` на `c++11`:

	$ git apply polus.patch

Опции:

    $ ./solver 
    $ Usage: ./solver Nx Ny K1 K2 Maxit Eps Tn Ll
    $ Where:
    $ Nx is positive int that represents grid hieght
    $ Ny is positive int that represents grid width
    $ K1 is positive int that represents square cells sequence length
    $ K2 is positive int that represents triangular cells sequence length
    $ Maxit is positive int that represents maximum iteration number
    $ Eps is positive float that represents accuracy
    $ Tn is tread numberLl is log level:
    $     <=0 - no logs
    $     >=1 - show time
    $     >=2 - show info
    $     >=3 - show arrays

Для измерения параллельного ускорения можно использовать готовый скрипт с
указанными размерностями, модифицируя его под свои нужды:

	$ ./run.sh 4000 4000 1

6) Сгенерировать портрет разреженной матрицы на основе сетки (вариант Б2),
посторить СЛАУ по полученному портрету и решить алгоритмом предобусловленного
метода CG, распараллелить средствами MPI.

Сборка:

	$ make

Для успешной работы на Полюсе можно использовать следующий патч, заменяющий
`clang++` на `g++` и `c++17` на `c++11`:

	$ git apply polus.patch

Опции:

    $ ./solver 
    $ Usage: ./solver Nx Ny K1 K2 Px Py Maxit Eps Tn Ll
    $ Where:
    $ Nx is positive int that represents grid hieght
    $ Ny is positive int that represents grid width
    $ K1 is positive int that represents square cells sequence length
    $ K2 is positive int that represents triangular cells sequence length
    $ Px is positive int that represents x axis decomposition param
    $ Py is positive int that represents y axis decomposition param
    $ Maxit is positive int that represents maximum iteration number
    $ Eps is positive float that represents accuracy
    $ Tn is tread numberLl is log level:
    $     <=0 - no logs
    $     >=1 - show time
    $     >=2 - show arrays
    $     >=3 - show info

Для измерения параллельного ускорения можно использовать готовый скрипт с
указанными размерностями, модифицируя его под свои нужды:

	$ ./run.sh

