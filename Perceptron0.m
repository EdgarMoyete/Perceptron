%Edgar Moises Hernandez-Gonzalez
%29/03/19-31/10/19
%Perceptron
%Actualiza la recta (se grafica todo en cada iteracion)

clear
clc

n = input('Ingrese n para las regiones A y B ejemplo 100: ');
punto1A = input('Ingrese el punto 1 para la region A ejemplo [0 0]: ');
punto2A = input('Ingrese el punto 2 para la region A ejemplo [4 4]: ');
punto1B = input('Ingrese el punto 1 para la region B ejemplo [4 4]: ');
punto2B = input('Ingrese el punto 2 para la region B ejemplo [8 8]: ');
intervalo = input('Ingrese el intervalo para los valores iniciales para los pesos ejemplo [0 1]: ');
COEF_APREN = input('Ingrese el valor del coeficiente de aprendizaje ejemplo 0.7: ');
numMaxItera = input('Ingrese numero maximo de iteraciones ejemplo 100 : ');
pregunta = input('¿Desea guardar en un archivo .mat los datos de las regiones generadas?, Si = 1 o No = 0: ');
a=intervalo(1);
b2=intervalo(2);

% si tuviera datos de un problema real solo cambie esta linea e importe sus
% datos en la variable datos donde las filas son la cantidad de datos y las
% columnas las variables del problema y la ultima columna deben ser
% etiquetas de 1 y -1 (la variable datos debe ser una matriz)
datos = aleatorio(n, punto1A, punto2A, punto1B, punto2B);

if pregunta == 1
    nombre = input('Nombre del archivo (escribirlo entre comillas simples): ');
    save(nombre, 'datos');
    disp(['Se guardo con exito ',nombre,'.mat']);
end

entradas = datos(:, 1:2);
salidasDeseadas = datos(:, 3);

x = entradas';
t = salidasDeseadas';
[f, c] = size(x); %f = #variables Xi...Xn y c = numero de datos
w = a + (b2 - a) * rand(1,f); %pesos en el intervalo [a, b]
b = rand; %bias
y = zeros(1, c);
net = zeros(1, c);
for epocas = 1:numMaxItera %epocas
    for i = 1:c
        net(i) = 0;
        for j = 1:f %sumatoria
            net(i) = net(i) + x(j, i) * w(1, j);
        end
        net(i) = net(i) - b; %net para pasarla a la funcion de activacion
        if (net(i) >= 0) %funcion de activacion escalon simetrico
            y(i) = 1;
        else
            y(i) = -1;
        end
        delta = t(i) - y(i); %regla delta
        for j = 1:f %ajuste de los pesos
            w(1, j) = w(1, j) + COEF_APREN * delta * x(j, i);
        end
        b = b - COEF_APREN * delta; %ajuste del bias
    end
    scatter(entradas(salidasDeseadas==1,1),entradas(salidasDeseadas==1,2));
    hold on;
    scatter(entradas(salidasDeseadas==-1,1),entradas(salidasDeseadas==-1,2));
    grid;
    xlabel('x');
    ylabel('y');
    axis([-10 10 -10 10]);
    tt=linspace(-10,10,21);
    exactitud = accuracy(t, y);
    recta = -(w(1)) / (w(2)) * tt + (b) / (w(2));
    plot(tt, recta);
    title(['Iteracion ', int2str(epocas), ' Exactitud: ', int2str(exactitud), '%']);
    disp(['Iteracion ', int2str(epocas), ' Exactitud: ', int2str(exactitud), '%']);
    hold off;
    if exactitud == 100
        break;
    end
    pause(0.5); %1 seg
end
disp('Pesos');
disp(w);
disp('sesgo');
disp(b);

disp('Despues del entrenamiento se evalua la red con el conjunto de pruebas');
disp('El conjunto de pruebas son datos que estan dentro de las regiones A y B');
disp('definidas en el entrenamiento pero son generados al azar ');
disp('y por lo tanto son diferentes al conjunto de entrenamiento');
exactitudPruebas = probarPerceptron(w, b, n, punto1A, punto2A, punto1B, punto2B);
disp('Exactitud con el conjunto de pruebas');
disp(exactitudPruebas);

%--------------------Funciones--------------------

function Puntos = aleatorio(N, puntoA1, puntoA2, puntoB1, puntoB2)
    A = zeros(N,3); %Puntos de la region A
    B = zeros(N,3); %Puntos de la region B
    Puntos = zeros(N*2,3);
    
    %disp('Region A (entre -10 y 10)')
    %puntoA1 = input('Ingrese el punto 1: ');    %El punto1 x,y P/e: [3 8]
    f = validaRango(puntoA1);   %Validar que el punto este en el rango 0=en rango, 1=fuera de rango
    while f   
        f = 0;  %Se reestablece la bandera y se vuelve a pedir el punto
        puntoA1 = input('Ingrese el punto 1: ');
        f = validaRango(puntoA1);
    end
    %puntoA2 = input('Ingrese el punto 2: ');    %El punto2 x,y P/e: [5 2]
    f = validaRango(puntoA2);   %Validar que el punto este en el rango 0=en rango, 1=fuera de rango
    while f   
        f = 0;  %Se reestablece la bandera y se vuelve a pedir el punto
        puntoA2 = input('Ingrese el punto 2: ');
        f = validaRango(puntoA2);
    end
    
    %disp('Region B (entre -10 y 10)')
    %puntoB1 = input('Ingrese el punto 1: ');    %El punto1 x,y P/e: [-1 3]
    f = validaRango(puntoB1);   %Validar que el punto este en el rango 0=en rango, 1=fuera de rango
    while f   
        f = 0;  %Se reestablece la bandera y se vuelve a pedir el punto
        puntoB1 = input('Ingrese el punto 1: ');
        f = validaRango(puntoB1);
    end
    %puntoB2 = input('Ingrese el punto 2: ');    %El punto2 x,y P/e: [-6 -4]
    f = validaRango(puntoB2);   %Validar que el punto este en el rango 0=en rango, 1=fuera de rango
    while f   
        f = 0;  %Se reestablece la bandera y se vuelve a pedir el punto
        puntoB2 = input('Ingrese el punto 2: ');
        f = validaRango(puntoB2);
    end
    
    %%%%%%%%%%%%    SE GENERARN LOS PUNTOS %%%%%%%%%%%%
    for i = 1:N
        %%%%    Componente x,y del punto i en region A    %%%
        xs1 = sort([puntoA1(1) puntoA2(1)]);a1 = xs1(1);b1 = xs1(2);
        A(i,1) = (b1-a1).*rand(1) + a1; %x aleatorio dentro area A
        ys1 = sort([puntoA1(2) puntoA2(2)]); a2 = ys1(1);b2 = ys1(2);
        A(i,2) = (b2-a2).*rand(1) + a2; %y aleatorio dentro area A
        A(i,3) = -1;
        
        %%%%    Componente x,y del punto i en region B    %%%
        xs2 = sort([puntoB1(1) puntoB2(1)]); a3 = xs2(1);b3 = xs2(2);
        B(i,1) = (b3-a3).*rand(1) + a3; %x aleatorio dentro area B
        ys2 = sort([puntoB1(2) puntoB2(2)]); a4 = ys2(1);b4 = ys2(2);
        B(i,2) = (b4-a4).*rand(1) + a4; %y aleatorio dentro area B
        B(i,3) = 1;
    end
    
    %%%%%%%%%   SE UNEN Y MEZCLAN LOS PUNTOS    %%%%%%%%%%%
    j=1;
    k=1;
    for i = 1:N*2
        if(rand(1)>.6)
            if(j<N+1)
                Puntos(i,1) = A(j,1);
                Puntos(i,2) = A(j,2);
                Puntos(i,3) = A(j,3);
                j = j + 1;
            else
                Puntos(i,1) = B (k,1);
                Puntos(i,2) = B (k,2);
                Puntos(i,3) = B (k,3);
                k = k + 1;
            end
        else
            if(k<N+1)
                Puntos(i,1) = B (k,1);
                Puntos(i,2) = B (k,2);
                Puntos(i,3) = B (k,3);
                k = k + 1;
            else
                Puntos(i,1) = A(j,1);
                Puntos(i,2) = A(j,2);
                Puntos(i,3) = A(j,3);
                j = j + 1;
            end
        end        
    end
    
%     A; %Puntos en area A
%     B; %Puntos en area B
%     Puntos; %Puntos mezclados
end

%Validar que las regiones A y B esten entre -10 y 10
function f = validaRango(Punto)
    if (Punto(1)<-10 || Punto(2)>10)
        f = 1;
    elseif ((Punto(2)<-10 || Punto(1)>10))
        f = 1;
    else
        f = 0;
    end
end

%Exactitud de la clasificacion en el entrenamiento
function exactitud = accuracy(t, y)
    correctos = 0;
    [f, c] = size(y); %filas y columnas
    for i = 1:c
        if (y(i) == t(i)) %si clasificacion correcta
            correctos = correctos + 1;
        end
    end
    exactitud = 100 * correctos / c;
end

%Probar el perceptron con un conjunto de pruebas
function exactitud = probarPerceptron(w, b, n, punto1A, punto2A, punto1B, punto2B)
    datos = aleatorio(n, punto1A, punto2A, punto1B, punto2B);
    entradas = datos(:, 1:2);
    salidasDeseadas = datos(:, 3);

    x = entradas';
    t = salidasDeseadas';
    [f, c] = size(x); %f = #variables Xi...Xn y c = numero de datos
    y = zeros(1, c);
    net = zeros(1, c);
    for i = 1:c
        net(i) = 0;
        for j = 1:f %sumatoria
            net(i) = net(i) + x(j, i) * w(1, j);
        end
        net(i) = net(i) - b; %net para pasarla a la funcion de activacion
        if (net(i) >= 0) %funcion de activacion escalon simetrico
            y(i) = 1;
        else
            y(i) = -1;
        end
    end
    figure;
    scatter(entradas(salidasDeseadas==1,1),entradas(salidasDeseadas==1,2))
    hold on
    scatter(entradas(salidasDeseadas==-1,1),entradas(salidasDeseadas==-1,2))
    grid
    xlabel('x')
    ylabel('y')
    axis([-10 10 -10 10]);
    tt=linspace(-10,10,21);
    exactitud = accuracy(t, y);
    recta = -(w(1)) / (w(2)) * tt + (b) / (w(2));
    plot(tt, recta);
    title(['Conjunto de Pruebas (Generalizacion)', ' Exactitud: ', int2str(exactitud), '%']);
    hold off
end