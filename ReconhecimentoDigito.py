import cv2
import numpy as np

def nothing(x):
    pass

#Variáveis
digits = cv2.imread('digitos.png', 0)
rows = np.vsplit(digits, 50)
cells = []

#Converter imagem em linhas de pixel
for row in rows:
    row_cell = np.hsplit(row, 50)
    for cell in row_cell:
        cell = cell.flatten() #converter imagem em uma linha de pixels
        cells.append(cell)

#Converter cells dpara float32
cells = np.array(cells, dtype = np.float32)

#range de 0 a 9 para a classificação de cada pacote de 250
k = np.arange(10)
cells_labels = np.repeat(k, 250)

#KNN
knn=cv2.ml.KNearest_create()
knn.train(cells, cv2.ml.ROW_SAMPLE, cells_labels)

#Execução da câmera
cap = cv2.VideoCapture(0)

#Criação da janela
cv2.namedWindow('trackbars')

#criação de trackbars
cv2.createTrackbar('x', 'trackbars', 100, 800, nothing)
cv2.createTrackbar('y', 'trackbars', 100, 800, nothing)
cv2.createTrackbar('w', 'trackbars', 100, 800, nothing)
cv2.createTrackbar('h', 'trackbars', 100, 800, nothing)
cv2.createTrackbar('th', 'trackbars', 100, 255, nothing)

while(True):

    #Leitura do frame
    ret, frame = cap.read()

    #Pega os valores das trackbars
    x = cv2.getTrackbarPos('x','trackbars')
    y = cv2.getTrackbarPos('y', 'trackbars')
    w = cv2.getTrackbarPos('w', 'trackbars')
    h = cv2.getTrackbarPos('h', 'trackbars')
    th = cv2.getTrackbarPos('th', 'trackbars')

    #roi do frame
    roi = frame[y:y+h, x:x+w]

    #Frame cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    #Threshold
    ret, threshold = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY_INV)

    #Dilate
    kernel = np.ones((5, 5), np.uint8)
    threshold = cv2.dilate(threshold, kernel, iterations = 5)

    #Encontrar dos contornos
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #For nos contornos
    for cnt in contours:

        #Digitos para teste
        test_cells=[]

       #Seta area do contorno em uma variavel
        area = cv2.contourArea(cnt)

        try:
            #Verifica se a area maior que 150
            if area > 800:

                #Tamanho do retângulo
                x_count,y_count,w_count,h_count = cv2.boundingRect(cnt)

                #Offset
                offset = 10

                #Crop
                crop= threshold[y_count-offset:y_count + h_count + offset, x_count-offset: x_count + w_count + offset]

                #Redimensionar

                if crop.shape[0] > crop.shape[1]:
                    newW = (crop.shape[1]*20)/crop.shape[0]
                    crop = cv2.resize(crop, (int(newW), 20))
                else:
                    newH = (crop.shape[0]*20)/crop.shape[1]
                    crop = cv2.resize(crop, (20, (newH)))

                cv2.imshow('crop',crop)

                #Aplicar retângulo
                cv2.rectangle(frame, (x_count, y_count), (x_count + w_count, y_count + h_count), (0,255,0), 2)

                #Largura e altura do crop
                height, width = crop.shape

                x3 = height if height > width else width
                y3 = height if height > width else width

                #Quadrado preto
                square = np.zeros((x3,y3), np.uint8)


                #Colocar o crop no meio do quadrado preto
                square[int((y3-height)/2):int(y3-(y3-height)/2), int((x3 - width)/2): int(x3-(x3-width)/2)] = crop

                test_cells.append(square.flatten())
                test_cells=np.array(test_cells, dtype = np.float32)

                #Teste/Predict
                ret, result, nighbours, dist = knn.findNearest(test_cells, k=1)

                #Print no frame do predict
                cv2.putText(frame,str(int(result[0][0])),(x,y-5), cv2.FONT_HERSHEY_SIMLPEX, 2, (0,255,0),2,cv2.LINE_AA)

        except:
            pass

    #Exibir frame com o crop
    cv2.imshow('threshold', threshold)

    #Exibir frame com o crop
    cv2.imshow('frame', frame)

    #waitkey
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

#Finalização
cap.release()
cv2.destroyAllWindows()
