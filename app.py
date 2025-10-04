# -*- coding: utf-8 -*-
import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
from pathlib import Path
import numpy as np 
from collections import Counter 
import time
# Adicionando a importação da PIL Image para a verificação de tipo (mantido)
from PIL import Image 
# IMPORTANTE: Adicionando 'io' para manipulação de bytes
import io 

# ===============================
# Configuração da página
# ===============================
st.set_page_config(
    page_title="Detecção de EPIs",
    page_icon="🦺",
    layout="wide"
)

# Inicializa o estado da sessão (mantido por consistência, mas o loop foi removido)
if 'camera_input_key' not in st.session_state:
    st.session_state.camera_input_key = 0

# ===============================
# Configuração Fixo de Confiança e Alerta Sonoro
# ===============================
conf_threshold = 0.50 # Valor fixo de confiança

# Áudio de bipe curto em Base64 (WAV simples, cerca de 0.1s de duração)
ALERT_SOUND_BASE64 = "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAAABkYXRhAAAAJAAA" 

# 1. HTML/JavaScript para CRIAR e expor a função de reprodução.
INITIAL_AUDIO_PLAYER_SETUP = f"""
<audio id="alert_audio_player" src="{ALERT_SOUND_BASE64}" preload="auto"></audio>
<script>
    // Função global para tocar o som de alerta de forma confiável
    function playAlertSound() {{
        const player = document.getElementById('alert_audio_player');
        if (player) {{
            // Pausa e reseta a posição para garantir que toque em loops rápidos
            player.pause();
            player.currentTime = 0;
            // Tenta tocar. Isso só funcionará após a interação inicial do usuário.
            player.play().catch(e => console.error("Erro ao tocar áudio:", e));
        }}
    }}
</script>
"""
# 2. HTML/JavaScript para CHAMAR a função de reprodução (Injetado APENAS quando o alerta for necessário)
HTML_ALERT_TRIGGER = """
    <script>
        if (typeof playAlertSound === 'function') {
            playAlertSound();
        }
    </script>
"""

# Injeta a configuração inicial do áudio globalmente
st.markdown(INITIAL_AUDIO_PLAYER_SETUP, unsafe_allow_html=True)


# ===============================
# Geração de Cores para as Classes
# ===============================
@st.cache_resource
def get_class_colors(num_classes=20):
    """Gera cores RGB distintas e fixas, otimizadas para contraste visual (usando HSV)."""
    np.random.seed(42) 
    hues = np.linspace(0, 179, num_classes, endpoint=False, dtype=np.uint8)
    np.random.shuffle(hues)
    hsv_colors = np.zeros((num_classes, 1, 3), dtype=np.uint8)
    hsv_colors[:, 0, 0] = hues
    hsv_colors[:, 0, 1] = 255
    hsv_colors[:, 0, 2] = 255
    rgb_colors_bgr = cv2.cvtColor(hsv_colors, cv2.COLOR_HSV2BGR)
    rgb_colors = cv2.cvtColor(rgb_colors_bgr, cv2.COLOR_BGR2RGB)
    return {i: tuple(rgb_colors[i, 0].tolist()) for i in range(num_classes)}

CLASS_COLORS = get_class_colors()

# ===============================
# Carregar modelo (Caminho AJUSTADO para a raiz do projeto)
# ===============================
@st.cache_resource
def load_model():
    """Carrega o modelo treinado a partir da raiz do projeto."""
    # Caminho ajustado para best.pt na raiz (C:\TCCWEB\best.pt)
    model_path = "best.pt" 
    if not os.path.exists(model_path):
        st.error(f"Erro: Arquivo do modelo '{model_path}' não encontrado. Certifique-se de que o arquivo 'best.pt' está na pasta raiz do seu projeto.")
        return None
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Erro ao carregar o modelo YOLO. Verifique se o arquivo está correto. Erro: {e}")
        return None

model = load_model()

# ===============================
# Sidebar e Mapeamento de Classes
# ===============================
st.sidebar.title("⚙️ Configurações")

# Mapeamento do nome amigável para o nome do modelo
label_map_model_to_friendly = {
    "helmet": "Capacete",
    "safety-vest": "Colete",
    "glasses": "Óculos",
    "gloves": "Luva",
    "shoes": "Bota",
    "ear-mufs": "Protetor Auricular", 
    "person": "Pessoa" 
}

# Lista de EPIs amigáveis para a Sidebar (excluindo "Pessoa")
epis_friendly = [name for name in label_map_model_to_friendly.values() if name != "Pessoa"]

# Emoji para o relatório
emoji_map = {
    "Capacete": "🪖",
    "Colete": "🦺",
    "Óculos": "🥽",
    "Luva": "🧤",
    "Bota": "👢",
    "Protetor Auricular": "🎧"
}

# Seleção de EPIs para monitorar na sidebar
selected_epis = st.sidebar.multiselect(
    "Selecione os EPIs para monitorar:",
    epis_friendly,
    default=epis_friendly
)

# ===============================
# Função Principal de Detecção (Adaptada para Frame ou Arquivo)
# ===============================

def process_detection(source, selected_epis):
    """
    Roda a inferência. Source é um objeto de arquivo (UploadedFile ou CameraInput).
    Lê o arquivo diretamente para memória.
    """
    if model is None:
        return None, [], set(), set(), set()
        
    result_img_rgb = None 
    temp_path = ""
    
    try:
        # 1. LER ARQUIVO PARA BYTES
        # Reseta o ponteiro do arquivo para o início, caso tenha sido lido antes
        source.seek(0)
        image_bytes = source.read()
        
        # 2. CONVERTER BYTES PARA ARRAY NUMPY USANDO OPENCV (melhor em cloud)
        # O numpy.frombuffer cria um array a partir dos bytes lidos
        np_arr = np.frombuffer(image_bytes, np.uint8)
        # O cv2.imdecode decodifica o array (que contém a imagem) para uma imagem OpenCV (BGR)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # 3. VERIFICAR FALHA NA DECODIFICAÇÃO
        if img_bgr is None:
            st.error("Erro: Não foi possível decodificar a imagem. O arquivo pode estar corrompido ou o formato é inválido.")
            return None, [], set(), set(), set()
            
        # Converter para RGB para exibição no Streamlit
        result_img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 4. SALVAR TEMPORARIAMENTE PARA A INFERÊNCIA DA ULTRALYTICS
        # Infelizmente, a função 'model()' da Ultralytics geralmente requer um caminho de arquivo.
        # Por isso, precisamos salvar rapidamente, mas garantimos que a leitura da imagem (o passo mais problemático)
        # já foi feito de forma robusta.
        file_extension = Path(source.name).suffix if source.name else ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
            tmp.write(image_bytes) # Escreve os bytes que já lemos
            temp_path = tmp.name
        
        # 5. Rodar inferência no arquivo temporário
        results = model(temp_path, conf=conf_threshold, save=False, verbose=False) 
    
    except Exception as e:
        # Este catch agora pega erros na leitura/decodificação OU na inferência
        st.error(f"Erro ao rodar o processamento (leitura/inferência). Erro: {e}")
        return None, [], set(), set(), set()
    finally:
        # Garante a limpeza do arquivo temporário
        if os.path.exists(temp_path):
            os.remove(temp_path) 
        
    
    detected_labels = []
    person_boxes = [] 
    
    # Desenhar manualmente e coletar detecções
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            eng_label = r.names[cls] 
            # A imagem para desenho é 'result_img_rgb' que foi criada a partir de img_bgr
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            pt_label = label_map_model_to_friendly.get(eng_label)
            
            # Rastreamento de Pessoas
            if eng_label == "person":
                person_boxes.append({'box': (x1, y1, x2, y2), 'label': pt_label})
                detected_labels.append(pt_label)
                continue

            # Ignora EPIs não mapeados ou não selecionados
            if pt_label is None or pt_label not in selected_epis:
                continue

            # Desenha bounding box para EPIs detectados (Cor da Classe)
            color_rgb = CLASS_COLORS.get(cls, (255, 255, 255)) 
            detected_labels.append(pt_label)

            cv2.rectangle(result_img_rgb, (x1, y1), (x2, y2), color_rgb, 2)
            cv2.putText(
                result_img_rgb,
                f"{pt_label}", 
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color_rgb, 
                2,
                cv2.LINE_AA
            )

    # Verificar atendidos x faltantes
    # Aqui, garantimos que "Pessoa" não conte como EPI na lógica de conformidade
    detected_set = set([label for label in detected_labels if label != 'Pessoa'])
    selected_set = set(selected_epis)
    # EPIs Atendidos (detectados E selecionados para monitoramento)
    atendidos = detected_set.intersection(selected_set)
    # EPIs Faltantes (selecionados para monitoramento E NÃO detectados)
    faltantes = selected_set - detected_set
    
    # ===============================
    # LÓGICA DE ALERTA VISUAL DE EPI FALTANTE (Vermelho na pessoa)
    # ===============================
    if faltantes and person_boxes:
        faltantes_str = ", ".join(sorted(list(faltantes)))
        alert_label = f"ALERTA! Faltando: {faltantes_str}"
        alert_color = (255, 0, 0) # Vermelho
        
        for person in person_boxes:
            x1, y1, x2, y2 = person['box']
            cv2.rectangle(result_img_rgb, (x1, y1), (x2, y2), alert_color, 4) 
            cv2.putText(
                result_img_rgb,
                alert_label, 
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                alert_color, 
                2,
                cv2.LINE_AA
            )
            
    elif person_boxes:
        # Se todos os EPIs selecionados foram encontrados (Verde na pessoa)
        safe_color = (0, 255, 0) # Verde
        for person in person_boxes:
            x1, y1, x2, y2 = person['box']
            cv2.rectangle(result_img_rgb, (x1, y1), (x2, y2), safe_color, 2)
            cv2.putText(
                result_img_rgb,
                "Pessoa (EPIs OK)", 
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                safe_color, 
                2,
                cv2.LINE_AA
            )


    # ==================================================
    # ADICIONAR RESUMO NA IMAGEM
    # ==================================================
    unique_detected_count = len(detected_set)
    if unique_detected_count > 0:
        unique_epis_names = ", ".join(sorted(list(detected_set)))
        summary_text = f"EPIs Detectados ({unique_detected_count}): {unique_epis_names}"
        text_x, text_y = 10, 35
        
        cv2.putText(result_img_rgb, summary_text, (text_x + 1, text_y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(result_img_rgb, summary_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    return result_img_rgb, person_boxes, atendidos, faltantes, detected_set

# ===============================
# Funções Auxiliares de Relatório
# ===============================
def generate_report_content(selected_epis, atendidos, faltantes, detected_set, person_boxes):
    """Gera o conteúdo do relatório no painel lateral."""
    
    st.markdown(f"**Pessoas detectadas:** {len(person_boxes)}")
    st.markdown(f"**EPIs Monitorados:** {len(selected_epis)}")
    st.markdown("---")

    # 1. Mensagem de alerta/sucesso
    alert_triggered = False
    if faltantes and person_boxes:
        st.error("⚠️ **ALERTA DE SEGURANÇA!** EPIs Faltantes. **(ALERTA SONORO DISPARADO)**")
        # DISPARO DO ÁUDIO
        st.markdown(HTML_ALERT_TRIGGER, unsafe_allow_html=True)
        alert_triggered = True
    elif person_boxes and not faltantes:
        st.success("✅ **CONFORME!** Todos os EPIs monitorados detectados.")
    elif not person_boxes and len(detected_set) > 0:
        st.info("EPIs detectados, mas nenhuma pessoa identificada para verificação de conformidade.")
    else:
        st.warning("Nenhuma pessoa ou EPI detectado na imagem.")

    st.markdown("---")
    
    # 2. Detalhamento dos EPIs
    st.markdown("**✅ EPIs em Conformidade:**")
    if atendidos:
        for epi in sorted(list(atendidos)):
            emoji = emoji_map.get(epi, "❓")
            st.markdown(f"{emoji} **{epi}** (Detectado)")
    else:
        st.markdown("*Nenhum EPI monitorado foi detectado.*")
    
    st.markdown("\n---")
    
    st.markdown("**❌ EPIs Faltantes:**")
    if faltantes:
        for epi in sorted(list(faltantes)):
            emoji = emoji_map.get(epi, "❓")
            st.markdown(f"{emoji} **{epi}** (Faltante/Não Detectado)")
    else:
        st.markdown("*Nenhum EPI faltante entre os monitorados.*")

# ===============================
# Área principal
# ===============================
st.title("🦺 Sistema de Detecção de EPIs")
st.write("Esta versão é otimizada para implantação em nuvem. Use 'Câmera (Snapshot)' para capturar fotos instantâneas.")
st.markdown("---")

# ----------------------------------------------------
# 1. SELEÇÃO DE ENTRADA (Apenas modos cloud-friendly)
# ----------------------------------------------------
input_mode = st.radio(
    "Selecione a fonte de entrada:",
    ("Câmera (Snapshot)", "Upload de Imagem"), 
    horizontal=True,
    index=0 
)

input_file = None

st.markdown("---")
st.subheader("2. Resultados da Detecção")
placeholder_col1, placeholder_col2 = st.columns([2, 1])

# ----------------------------------------------------
# 2. PROCESSAMENTO (Snapshot ou Upload)
# ----------------------------------------------------

with placeholder_col1:
    if input_mode == "Câmera (Snapshot)":
        st.subheader("1. Câmera (Snapshot)")
        # st.camera_input é a maneira Cloud-Safe de acessar a câmera
        input_file = st.camera_input(
            "Tire uma foto para análise:",
            key=st.session_state.camera_input_key
        )
    else:
        st.subheader("1. Upload de Imagem")
        input_file = st.file_uploader(
            "Escolha uma imagem para upload:",
            type=["jpg", "jpeg", "png"]
        )

# Executa a detecção se houver um arquivo de entrada
if input_file is not None:
    
    # Roda o processamento
    processed_img_rgb, person_boxes, atendidos, faltantes, detected_set = process_detection(
        input_file, 
        selected_epis
    )
    
    # === CORREÇÃO CRÍTICA DO FLUXO DE RENDERIZAÇÃO ===
    # Agora, todo o bloco de exibição de resultados (imagem, download e relatório)
    # é executado apenas se a imagem foi processada com sucesso (array NumPy válido).
    if processed_img_rgb is not None and isinstance(processed_img_rgb, np.ndarray):
        
        with placeholder_col1:
            # LINHA QUE CAUSAVA ERRO AGORA ESTÁ PROTEGIDA
            st.image(processed_img_rgb, use_container_width=True, caption="Resultado da Detecção com Alerta")

            # Botão de download
            result_img_bgr_dl = cv2.cvtColor(processed_img_rgb, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode(".jpg", result_img_bgr_dl)
            st.download_button(
                label="📥 Baixar imagem com detecção",
                data=buffer.tobytes(),
                file_name="resultado.jpg",
                mime="image/jpeg"
            )

        with placeholder_col2:
            st.subheader("📊 Relatório")
            # O relatório agora é gerado apenas se o processamento for bem-sucedido
            generate_report_content(selected_epis, atendidos, faltantes, detected_set, person_boxes)
            
    else:
        # Exibe uma mensagem de status se o processamento falhar ou retornar None
        with placeholder_col1:
            st.info("Aguardando processamento. Verifique se há mensagens de erro acima, caso o resultado não apareça.")
