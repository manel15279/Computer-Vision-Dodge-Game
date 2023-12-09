import cv2
import numpy as np
import math

#==========================Myfunctions==================================
import numpy as np


def my_sum(seq):
    total = 0

    # Convertir le générateur en liste
    seq_list = list(seq)

    # Utiliser la longueur de la liste
    length = len(seq_list)

    index = 0
    while index < length:
        total += seq_list[index]
        index += 1

    return total


def my_max(arg1, arg2):
    seq = [arg1, arg2]
    
    if not seq:
        # Gérer le cas où la séquence est vide
        raise ValueError("my_max() arg is an empty sequence")

    current_max = float('-inf')  # initialiser à l'infini négatif
    index = 0
    length = len(seq)

    while index < length:
        if seq[index] > current_max:
            current_max = seq[index]
        index += 1

    return current_max


def my_min(arg1, arg2):
    seq = [arg1, arg2]

    if not seq:
        # Gérer le cas où la séquence est vide
        raise ValueError("my_min() arg is an empty sequence")

    current_min = float('inf')  # initialiser à l'infini positif
    index = 0
    length = len(seq)

    while index < length:
        if seq[index] < current_min:
            current_min = seq[index]
        index += 1

    return current_min


def my_abs(num):
    if num < 0:
        return -num
    return num

def my_max2(seq):
    if not seq:
        # Gérer le cas où la séquence est vide
        raise ValueError("my_max() arg is an empty sequence")

    current_max = float('-inf')  # initialiser à l'infini négatif
    index = 0
    length = len(seq)

    while index < length:
        if seq[index] > current_max:
            current_max = seq[index]
        index += 1

    return current_max


def my_min2(seq):
    if not seq:
        # Gérer le cas où la séquence est vide
        raise ValueError("my_min() arg is an empty sequence")

    current_min = float('inf')  # initialiser à l'infini positif
    index = 0
    length = len(seq)

    while index < length:
        if seq[index] < current_min:
            current_min = seq[index]
        index += 1

    return current_min




def dilation(image, kernel):
    rows, cols = image.shape
    result = np.zeros(image.shape, image.dtype)

    k_rows = len(kernel)
    k_cols = len(kernel[0])

    i = 1
    while i < rows - 1:
        j = 1
        while j < cols - 1:
            # Initial value is set to 255 (white)
            min_val = 0

            m = 0
            while m < k_rows:
                n = 0
                while n < k_cols:
                    # Multiply corresponding pixel value with the kernel value
                    pixel_value = image[i - 1 + m, j - 1 + n] * kernel[m][n]
                    # Find the minimum value
                    min_val = my_max(min_val, pixel_value)

                    n += 1 

                m += 1 

            result[i, j] = min_val

            j += 1 

        i += 1  

    return result



def erode(mask,kernel):
    y=0
    ym,xm=kernel.shape
    m=int((xm-1)/2)
    mask2=255*np.ones(mask.shape,mask.dtype)
    while y<mask.shape[0]:
        x=0
        while x<mask.shape[1]:
            
             if  not( y<m or y>(mask.shape[0]-1-m) or x<m or x>(mask.shape[1]-1-m)):
                v=mask[y-m:y+m+1,x-m:x+m+1] 
           
                h=0
                while(h<ym):
                     w=0
                     while(w<xm): 
                        if(v[h,w]<kernel[h,w]):
                             mask2[y,x]=0
                             break
                        w+=1
                     if(mask2[y,x]==0): 
                         break
                     h+=1
             x+=1
        y+=1

    return mask2

#=======================================================================
def mean_filter(img, kernel_size):
    rows, cols, channels = img.shape
    filtered_img = np.zeros(img.shape, img.dtype)

    half_kernel = kernel_size // 2

    i = half_kernel
    while i < rows - half_kernel:
        j = half_kernel
        while j < cols - half_kernel:
            k = 0
            while k < channels: # traiter le cas des image GRAYSCALE ou COLOR
                # Calculer la moyenne dans la fenêtre du noyau
                sum_pixels = 0
                m = -half_kernel
                while m <= half_kernel:
                    n = -half_kernel
                    while n <= half_kernel:
                        sum_pixels += img[i + m, j + n, k]
                        n += 1
                    m += 1

                # Mettre à jour la valeur du pixel avec la moyenne calculée
                filtered_img[i, j, k] = sum_pixels // (kernel_size ** 2)
                k += 1

            j += 1

        i += 1

    return filtered_img




#========================TRI===========================
def heapify(arr, n, i):
    largest = i
    left_child = 2 * i + 1
    right_child = 2 * i + 2

    while left_child < n or right_child < n:
        if left_child < n and arr[left_child] > arr[largest]:
            largest = left_child

        if right_child < n and arr[right_child] > arr[largest]:
            largest = right_child

        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            i = largest
            left_child = 2 * i + 1
            right_child = 2 * i + 2
        else:
            break

def heap_sort(arr):
    n = len(arr)

    # Construire un tas (heapify) en partant du dernier nœud non feuille
    i = n // 2 - 1
    while i >= 0:
        heapify(arr, n, i)
        i -= 1

    # Extraire les éléments un par un du tas
    i = n - 1
    while i > 0:
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)
        i -= 1

#===============================================================
def median_filter(img, kernel_size):
    rows, cols, channels = img.shape
    filtered_img = np.zeros(img.shape, img.dtype)

    half_kernel = kernel_size // 2

    i = half_kernel
    while i < rows - half_kernel:
        j = half_kernel
        while j < cols - half_kernel:
            k = 0
            while k < channels:   # traiter le cas des image GRAYSCALE ou COLOR
                # Collecter les valeurs des pixels dans la fenêtre du noyau
                values = []
                m = -half_kernel
                while m <= half_kernel:
                    n = -half_kernel
                    while n <= half_kernel:
                        values.append(img[i + m, j + n, k])
                        n += 1
                    m += 1

                # Calculer la médiane des valeurs collectées
                heap_sort(values)
                median_value = values[len(values) // 2]

                # Mettre à jour la valeur du pixel avec la médiane calculée
                filtered_img[i, j, k] = median_value
                k += 1

            j += 1

        i += 1

    return filtered_img



def gradient_filter(img, kernel_size):
    rows, cols, channels = img.shape
    gradient_img = np.zeros(img.shape, img.dtype)

    half_kernel = kernel_size // 2

    i = half_kernel
    while i < rows - half_kernel:
        j = half_kernel
        while j < cols - half_kernel:
            k = 0
            while k < channels:
                # Collecter les valeurs des pixels dans la fenêtre du noyau
                values = []
                m = -half_kernel
                while m <= half_kernel:
                    n = -half_kernel
                    while n <= half_kernel:
                        values.append(img[i + m, j + n, k])
                        n += 1
                    m += 1

                # Calculer le gradient en utilisant les valeurs collectées
                gradient_value = my_max2(values) - my_min2(values)

                # Mettre à jour la valeur du pixel avec le gradient calculé
                gradient_img[i, j, k] = gradient_value
                k += 1

            j += 1

        i += 1

    return gradient_img



def gaussian(x, y, sigma):
    return (1.0 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

def generate_gaussian_kernel(size, sigma):
    half_size = size // 2
    kernel = []

    i = -half_size
    while i <= half_size:
        row = []
        j = -half_size
        while j <= half_size:
            row.append(gaussian(i, j, sigma))
            j += 1
        kernel.append(row)
        i += 1

    # Normalize the kernel
    kernel_sum = my_sum(my_sum(row) for row in kernel)
    normalized_kernel = [[element / kernel_sum for element in row] for row in kernel]

    return normalized_kernel

def convolve(image, kernel):
    rows, cols, channels = image.shape
    filtered_img = np.zeros(image.shape, image.dtype)

    half_kernel = len(kernel) // 2

    i = half_kernel
    while i < rows - half_kernel:
        j = half_kernel
        while j < cols - half_kernel:
            k = 0
            while k < channels:
                # Calculate the weighted sum in the kernel window
                sum_pixels = 0
                m = -half_kernel
                while m <= half_kernel:
                    n = -half_kernel
                    while n <= half_kernel:
                        sum_pixels += image[i + m, j + n, k] * kernel[m + half_kernel][n + half_kernel]
                        n += 1
                    m += 1

                # Update the pixel value with the weighted sum
                filtered_img[i, j, k] = int(sum_pixels)
                k += 1

            j += 1

        i += 1

    return filtered_img



def filtre_laplacien(image):
    # Définir le noyau du filtre laplacien
    kernel = [[0, 1, 0],
              [1, -4, 1],
              [0, 1, 0]]

    # Obtenir la taille de l'image
    rows, cols = len(image), len(image[0])

    # Initialiser une liste pour le résultat avec une boucle while
    resultat = []
    i = 0
    while i < rows:
        j = 0
        row = []
        while j < cols:
            row.append(0)
            j += 1
        resultat.append(row)
        i += 1

    # Appliquer la convolution manuellement avec une boucle while
    i = 1
    while i < rows - 1:
        j = 1
        while j < cols - 1:
            roi = [row[j-1:j+2] for row in image[i-1:i+2]]
            resultat[i][j] = my_sum(roi_val * kernel_val for roi_row, kernel_row in zip(roi, kernel) for roi_val, kernel_val in zip(roi_row, kernel_row))
            j += 1
        i += 1

    # Convertir le résultat en valeurs absolues et en entier 8 bits
    resultat_uint8 = [[my_min(255, my_max(0, int(my_abs(val)))) for val in row] for row in resultat]

    return np.array(resultat_uint8, dtype=np.uint8)


def erode(mask,kernel):
    y=0
    ym,xm=kernel.shape
    m=int((xm-1)/2)
    mask2=255*np.ones(mask.shape,mask.dtype)
    while y<mask.shape[0]:
        x=0
        while x<mask.shape[1]:
            
             if  not( y<m or y>(mask.shape[0]-1-m) or x<m or x>(mask.shape[1]-1-m)):
                v=mask[y-m:y+m+1,x-m:x+m+1] 
           
                h=0
                while(h<ym):
                     w=0
                     while(w<xm): 
                        if(v[h,w]<kernel[h,w]):
                             mask2[y,x]=0
                             break
                        w+=1
                     if(mask2[y,x]==0): 
                         break
                     h+=1
             x+=1
        y+=1

    return mask2


def dilation(image, kernel):
    rows, cols = image.shape
    result = np.zeros(image.shape, image.dtype)


    k_rows = len(kernel)
    k_cols = len(kernel[0])


    i = 1
    while i < rows - 1:
        j = 1
        while j < cols - 1:
            # Initial value is set to 255 (white)
            min_val = 0

            m = 0
            while m < k_rows:
                n = 0
                while n < k_cols:
                    # Multiply corresponding pixel value with the kernel value
                    pixel_value = image[i - 1 + m, j - 1 + n] * kernel[m][n]
                    # Find the minimum value
                    min_val = my_max(min_val, pixel_value)

                    n += 1 

                m += 1 

            result[i, j] = min_val

            j += 1 

        i += 1  

    return result

def seuillage_binaire(image_path, seuil):
    # Charger l'image en niveaux de gris
    image_gris = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Appliquer le seuillage binaire
    pixels_seuilles = (image_gris > seuil) * 255
    result_affichage = np.uint8(pixels_seuilles)

    return result_affichage  # Retourner result_affichage, pas pixels_seuilles



def closing(image, kernel):
    # Apply erosion first
    eroded_image = erode(image, kernel)
    # Apply dilation on the eroded image
    closed_image = dilation(eroded_image, kernel)
    return closed_image


def opening(image, kernel):
    # Apply dilation first
    dilated_image = dilation(image, kernel)
    # Apply erosion on the dilated image
    opened_image = erode(dilated_image, kernel)
    return opened_image



def prewitt_filter(image, direction='horizontal'):
    # Define Prewitt filters
    if direction == 'horizontal':
        kernel = np.array([[-1, 0, 1],
                           [-1, 0, 1],
                           [-1, 0, 1]])
    elif direction == 'vertical':
        kernel = np.array([[-1, -1, -1],
                           [0, 0, 0],
                           [1, 1, 1]])
    else:
        raise ValueError("Invalid direction. Use 'horizontal' or 'vertical'.")

    # Get image dimensions
    rows, cols = image.shape

    # Initialize result image
    result = np.zeros(image.shape, dtype=np.float32)

    # Initialize loop variables
    i = 1
    while i < rows - 1:
        j = 1
        while j < cols - 1:
            # Apply convolution
            result[i, j] = np.sum(image[i-1:i+2, j-1:j+2] * kernel)

            # Increment column index
            j += 1

        # Increment row index
        i += 1

    # Ensure result is in the valid range [0, 255]
    result[result < 0] = 0
    result[result > 255] = 255


    # Convert to uint8 (8-bit image)
    result = np.uint8(result)
    # result = result % 256

    return result



def sobel(image, kernel):
    # Taille de l'image et du noyau
    img_height, img_width = image.shape
    kernel_size = len(kernel)

    # Demi-taille du noyau pour le padding
    kernel_half = kernel_size // 2

    # Image résultante de la convolution
    result = np.zeros_like(image, dtype=np.float64)

    # Appliquer la convolution
    y = kernel_half
    while y < img_height - kernel_half:
        x = kernel_half
        while x < img_width - kernel_half:
            roi = image[y - kernel_half:y + kernel_half + 1, x - kernel_half:x + kernel_half + 1]
            result[y, x] = np.sum(roi * kernel)
            x += 1
        y += 1

    return result
