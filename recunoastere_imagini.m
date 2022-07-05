%% Imagini care contin diferite variatii ale literei S

A = imread('s1.png'); %citeste imaginea
B1 = rgb2gray(A); %conversie in gray
B1 = reshape(B1.',1,[]); %conversie din matrice in vector linie
A = imread('s2.png'); 
B2 = rgb2gray(A); 
B2 = reshape(B2.',1,[]); 
A = imread('s3.png'); 
B3 = rgb2gray(A); 
B3 = reshape(B3.',1,[]); 
A = imread('s4.png'); 
B4 = rgb2gray(A); 
B4 = reshape(B4.',1,[]);
A = imread('s5.png'); 
B5 = rgb2gray(A); 
B5 = reshape(B5.',1,[]);
A = imread('s6.png'); 
B6 = rgb2gray(A); 
B6 = reshape(B6.',1,[]);
A = imread('s7.png'); 
B7 = rgb2gray(A); 
B7 = reshape(B7.',1,[]);
A = imread('s8.png'); 
B8 = rgb2gray(A); 
B8 = reshape(B8.',1,[]);
A = imread('s9.png'); 
B9 = rgb2gray(A); 
B9 = reshape(B9.',1,[]);
A = imread('s10.png'); 
B10 = rgb2gray(A); 
B10 = reshape(B10.',1,[]);
A = imread('s11.png'); 
B11 = rgb2gray(A); 
B11 = reshape(B11.',1,[]);
A = imread('s12.png'); 
B12 = rgb2gray(A); 
B12 = reshape(B12.',1,[]);
A = imread('s13.png'); 
B13 = rgb2gray(A); 
B13 = reshape(B13.',1,[]);
C1=[B1;B2;B3;B4;B5;B6;B7;B8;B9;B10;B11;B12;B13];
C1=double(C1);

%% Imagini random diferite de S

A = imread('n1.png'); 
B1 = rgb2gray(A); 
B1 = reshape(B1.',1,[]);
A = imread('n2.png'); 
B2 = rgb2gray(A); 
B2 = reshape(B2.',1,[]);
A = imread('n3.png'); 
B3 = rgb2gray(A); 
B3 = reshape(B3.',1,[]);
A = imread('n4.png'); 
B4 = rgb2gray(A); 
B4 = reshape(B4.',1,[]);
A = imread('n5.png'); 
B5 = rgb2gray(A); 
B5 = reshape(B5.',1,[]);
A = imread('n6.png'); 
B6 = rgb2gray(A); 
B6 = reshape(B6.',1,[]);
A = imread('n7.png'); 
B7 = rgb2gray(A); 
B7 = reshape(B7.',1,[]);
A = imread('n8.png'); 
B8 = rgb2gray(A); 
B8 = reshape(B8.',1,[]);
A = imread('n9.png'); 
B9 = rgb2gray(A); 
B9 = reshape(B9.',1,[]);
A = imread('n10.png'); 
B10 = rgb2gray(A); 
B10 = reshape(B10.',1,[]);
A = imread('n11.png'); 
B11 = rgb2gray(A); 
B11 = reshape(B11.',1,[]);
A = imread('n12.png'); 
B12 = rgb2gray(A); 
B12 = reshape(B12.',1,[]);
A = imread('n13.png'); 
B13 = rgb2gray(A); 
B13 = reshape(B13.',1,[]);


C2=[B1;B2;B3;B4;B5;B6;B7;B8;B9;B10;B11;B12;B13];
C2=double(C2);


%% CVX

cvx_begin
variable w(100);
variable b(1);
minimize ((1/2)*w'*w)
subject to
C1*w-b*ones(13,1)>=ones(13,1); %1*C1*.. >=
C2*w-b*ones(13,1)<=-ones(13,1); % -1*C2*... >=  
cvx_end

%% Testare
disp 'Test CVX'

A=imread('t1.png');
A=rgb2gray(A);
A = reshape(A.',1,[]);
A=double(A);
rez1 = w'*A'-b;
if (rez1 > 0)
    disp 'Prima imagine test contine S'
else
    disp 'Prima imagine test nu contine S'
end

A=imread('t2.png');
A=rgb2gray(A);
A = reshape(A.',1,[]);
A=double(A);
rez2 = w'*A'-b;
if (rez2 > 0)
    disp 'A doua imagine test contine S'
else
    disp 'A doua imagine test nu contine S'
end

%% Regresie logistica

n=26; %nr de poze
C1=C1'; %aduc informatia despre o poza de la forma linie la forma coloana
C2=C2';
pixeli=[C2 C1;ones(1,26)]; %matricea cu pixeli
y=[zeros(13,1);ones(13,1)]; %etichetele
[m,~]=size(pixeli); %nr parametrilor de regresie
epsilon = 0.001; % marja eroarea 
maxIter = 100000; % nr maxim de iteratii
wr=zeros(m,1); %w gasit cu regresie logistica

%Metoda Gradient

%Calculam gradientul functiei
F=@(wr)(1/n)*(-y'*log(sigmoid(pixeli'*wr))-(ones(n,1)-y)'*log(1-sigmoid(pixeli'*wr)));
gradient=(1/n)*pixeli*(sigmoid(pixeli'*wr)-y);

%Metoda Gradient cu pas ideal
k=0;
norma=norm(gradient);
norm_init=norma;
ngpi=[]; %vectorul cu norme pt metoda gradient cu pas constant
ngpi=[ngpi;norma];
while (norma>=epsilon && k<maxIter)
    cost=@(alfa) F(wr-alfa*gradient);
    alfa=fminbnd(cost,0,1); %pasul ideal
    wr=wr-alfa*gradient;
    gradient=(1/n)*pixeli*(sigmoid(pixeli'*wr)-y);
    norma=norm(gradient);
    ngpi=[ngpi;norma];
    k=k+1;
end
wgpi=wr; %salvam w pt metoda gradient pas ideal

%% Testare Regresie logistica

disp 'Test Regresie Logistica'
prag_decizie=0.5;
A=imread('t3.png');
A=rgb2gray(A);
A = reshape(A.',1,[]);
A=double(A);
A=[A 1];
if (sigmoid(wr'*A') >= prag_decizie)
    disp 'Imaginea contine S';
else
    disp 'Imaginea nu contine S';
end
A=imread('t4.png');
A=rgb2gray(A);
A = reshape(A.',1,[]);
A=double(A);
A=[A 1];
if (sigmoid(wr'*A') >= prag_decizie)
    disp 'Imaginea contine S';
else
    disp 'Imaginea nu contine S';
end

%% Plotare
figure();
plot(ngpi, 'r', 'LineWidth', 2);

