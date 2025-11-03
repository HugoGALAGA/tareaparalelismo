'''
@tortolala
Pokemon image processing pipeline.
'''

from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from pika_banner import print_pikachu
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import requests
import time
import os

def download_single_image(pokemon_id, dir_name='pokemon_dataset', base_url='https://raw.githubusercontent.com/HybridShivam/Pokemon/master/assets/imagesHQ'):
    """
    Descarga la imagen para un único ID de Pokémon.
    Esta es la función que cada hilo ejecutará.
    """
    file_name = f'{pokemon_id:03d}.png'
    url = f'{base_url}/{file_name}'
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        img_path = os.path.join(dir_name, file_name)
        with open(img_path, 'wb') as f:
            f.write(response.content)
            
    except requests.exceptions.RequestException as e:
        return f'  Error descargando {file_name}: {e}'
    return None 

def download_pokemon(n=150, dir_name='pokemon_dataset'):
    '''
    Descarga las imágenes de los primeros n Pokemones de forma concurrente.
    '''
    os.makedirs(dir_name, exist_ok=True)
    
    print(f'\nDescargando {n} pokemones (concurrente)...\n')
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        pokemon_ids = range(1, n + 1)
        
        results = list(tqdm(executor.map(download_single_image, pokemon_ids), total=n, desc='Descargando', unit='img'))

    for error in results:
        if error:
            tqdm.write(error)

    total_time = time.time() - start_time
    print(f'  Descarga completada en {total_time:.2f} segundos')
    print(f'  Promedio: {total_time/n:.2f} s/img')
    
    return total_time

def process_single_image(image_file, dir_origin='pokemon_dataset', dir_name='pokemon_processed'):
    """Aplica transformaciones a una única imagen de Pokémon."""
    try:
        path_origin = os.path.join(dir_origin, image_file)
        img = Image.open(path_origin).convert('RGB')
        
        img = img.filter(ImageFilter.GaussianBlur(radius=10))
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)
        img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
        img_inv = ImageOps.invert(img)
        img_inv = img_inv.filter(ImageFilter.GaussianBlur(radius=5))
        width, height = img_inv.size
        img_inv = img_inv.resize((width * 2, height * 2), Image.LANCZOS)
        img_inv = img_inv.resize((width, height), Image.LANCZOS)
    
        saving_path = os.path.join(dir_name, image_file)
        img_inv.save(saving_path, quality=95)
        return None 
    except Exception as e:
        return f'  Error procesando {image_file}: {e}' 
        
def process_pokemon(dir_origin='pokemon_dataset', dir_name='pokemon_processed'):
    '''
    Procesa las imágenes en paralelo aplicando múltiples transformaciones.
    '''
    os.makedirs(dir_name, exist_ok=True)
    images = sorted([f for f in os.listdir(dir_origin) if f.endswith('.png')])
    total = len(images)
    
    print(f'\nProcesando {total} imágenes (paralelo)...\n')
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(process_single_image, images), total=total, desc='Procesando', unit='img'))

    for error in results:
        if error:
            tqdm.write(error)

    total_time = time.time() - start_time
    print(f'  Procesamiento completado en {total_time:.2f} segundos')
    print(f'  Promedio: {total_time/total:.2f} s/img\n')
    
    return total_time


if __name__ == '__main__':

    print('='*60)
    print_pikachu()
    print('   POKEMON IMAGE PROCESSING PIPELINE')
    print('='*60)
    
    download_time = download_pokemon()
    
    processing_time = process_pokemon()
    
    total_time = download_time + processing_time

    print('='*60)
    print('RESUMEN DE TIEMPOS\n')
    print(f'  Descarga:        {download_time:.2f} seg')
    print(f'  Procesamiento:   {processing_time:.2f} seg\n')
    print(f'  Total:           {total_time:.2f} seg')
    print('='*60)
