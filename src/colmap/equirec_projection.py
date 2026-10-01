import numpy as np
import cv2

class FlexibleEquirecProjection:
    def __init__(self, equirec_rgb_path:str, equirec_depth_path:str=None, equirec_mask_path:str=None, kernel_size_mask:int=3):
        """
        Projetor flexível com suporte a matriz de alinhamento 3x3.
        """
        self.depth_radial = None
        self.img_mask = None

        # 1. Carregar Imagem RGB
        self.img_rgb = cv2.imread(equirec_rgb_path)
        self.H, self.W, _ = self.img_rgb.shape
        
        # 2. Carregar e Decodificar Profundidade Inicial
        if equirec_depth_path:
            img_d_raw = cv2.imread(equirec_depth_path)
            rgb_ordered = cv2.cvtColor(img_d_raw, cv2.COLOR_BGR2RGB)
            data = rgb_ordered.astype(np.float64)
            self.depth_radial = (data[:, :, 0] + data[:, :, 1] * 256.0 + data[:, :, 2] * 65536.0) / 10000.0
                
            # [NOVO] 3. Filtragem Estatística dos Limites do LiDAR
            # Criamos uma máscara booleana isolando apenas o que está no intervalo métrico correto
            lidar_valido_mask = (self.depth_radial >= 0.0) & (self.depth_radial <= 1000.0)
            
            # IMPORTANTE: Zeramos os valores inválidos na fotosfera base antes do remap.
            # Isso evita que valores absurdos (negativos ou gigantes) "vazem" para os pixels bons na interpolação
            self.depth_radial[~lidar_valido_mask] = 0.0

        # [NOVO] 4. Integração e Tratamento da Máscara Booleana + Mediana 3x3
        if equirec_mask_path:
            img_mask_raw = cv2.imread(equirec_mask_path, cv2.IMREAD_GRAYSCALE)
            if img_mask_raw.shape[:2] != (self.H, self.W):
                img_mask_raw = cv2.resize(img_mask_raw, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
                
            # Combinamos a sua máscara original (True onde > 127) COM o filtro de limites do LiDAR
            mascara_combinada_bool = (img_mask_raw > 127) & lidar_valido_mask
            
            # Convertemos para uint8 para aplicar o filtro da mediana 3x3 do OpenCV de forma ultra rápida
            mask_uint8 = mascara_combinada_bool.astype(np.uint8) * 255
            self.img_mask = cv2.medianBlur(mask_uint8, kernel_size_mask) # Limpa pixels espúrios isolados
            
    def encode_24bit_depth(self, depth_m):
        depth_scaled = np.clip(depth_m * 10000.0, 0, 16777215).astype(np.uint32)
        r = (depth_scaled & 0xFF).astype(np.uint8)
        g = ((depth_scaled >> 8) & 0xFF).astype(np.uint8)
        b = ((depth_scaled >> 16) & 0xFF).astype(np.uint8)
        return np.stack((r, g, b), axis=-1)
    
    def generate_virtual_face(self, theta_center_deg, cubemap_alignment=None, fov_h_deg=30.0, fov_v_deg=60.0, face_w=512, face_h=512):
        """
        Gera uma visada projetiva aplicando a máscara perfeitamente limpa e filtrada.
        """
        theta_c = np.radians(theta_center_deg)
        
        # Grade de projeção da câmera virtual
        x = np.linspace(-np.tan(np.radians(fov_h_deg / 2)), np.tan(np.radians(fov_h_deg / 2)), face_w)
        y = np.linspace(-np.tan(np.radians(fov_v_deg / 2)), np.tan(np.radians(fov_v_deg / 2)), face_h)
        xx, yy = np.meshgrid(x, y)
        yy = -yy  
        
        X = xx * np.cos(theta_c) + np.sin(theta_c)
        Z = -xx * np.sin(theta_c) + np.cos(theta_c)
        Y = yy
        
        rays = np.stack([X, Y, Z], axis=0)
        
        if cubemap_alignment is not None:
            rays = np.einsum('ij,jkl->ikl', cubemap_alignment, rays)
            X, Y, Z = rays[0], rays[1], rays[2]
        
        norm = np.sqrt(X**2 + Y**2 + Z**2)
        
        theta = np.arctan2(X, Z)
        phi = np.arcsin(Y / norm)
        
        map_x = (theta + np.pi) / (2 * np.pi) * (self.W - 1)
        map_y = (np.pi / 2 - phi) / np.pi * (self.H - 1)
        
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        
        # Remapeamento das matrizes pré-filtradas
        face_rgb = cv2.remap(self.img_rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        try:
            # Remapeamento das matrizes pré-filtradas
            face_d_radial = cv2.remap(self.depth_radial, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
            
            # A máscara usa INTER_NEAREST para manter o corte binário perfeito e seco nas bordas
            face_mask = cv2.remap(self.img_mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            # Correção Planar 
            face_d_planar = face_d_radial / norm
            # Aplicação final do filtro: Tudo o que foi invalidado (pela máscara original, pela mediana ou pelo limite do LiDAR) vira 0.0 metros
            invalid_pixels = face_mask < 127
            face_d_planar[invalid_pixels] = 0.0
            
            face_d_rgb = self.encode_24bit_depth(face_d_planar)
            face_d_encoded_bgr = cv2.cvtColor(face_d_rgb, cv2.COLOR_RGB2BGR)
        except:
            face_d_encoded_bgr, face_mask = None, None

        return face_rgb, face_d_encoded_bgr, face_mask #face_d_planar