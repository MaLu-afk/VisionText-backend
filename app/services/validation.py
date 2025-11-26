import re
from typing import List, Set, Tuple
from app.config import settings


class ContentValidationService:
    def __init__(self):
        self.banned_words = self._load_banned_words()
    
    def _load_banned_words(self) -> Set[str]:
        """Cargar lista de palabras prohibidas"""
        banned_words = {            
            # ==================== CONTENIDO VIOLENTO/PELIGROSO ====================
            # Español
            'asesinato', 'asesinar', 'matar', 'muerte', 'muerto', 'arma', 'armas',
            'pistola', 'revólver', 'fusil', 'escopeta', 'cuchillo', 'navaja',
            'sangre', 'sangriento', 'sangrienta', 'violencia', 'violento', 'violenta',
            'gore', 'terrorismo', 'terrorista', 'bomba', 'explosivo', 'disparar',
            'pelea', 'peleas', 'agresión', 'agredir', 'golpear', 'apuñalar',
            'suicidio', 'suicidarse', 'matarse', 'acoso', 'acosar', 'abusar',
            
            # Inglés
            'murder', 'kill', 'killing', 'death', 'dead', 'weapon', 'weapons',
            'gun', 'pistol', 'revolver', 'rifle', 'shotgun', 'knife', 'blade',
            'blood', 'bloody', 'violence', 'violent', 'gore', 'terrorism', 'terrorist',
            'bomb', 'explosive', 'shoot', 'shooting', 'fight', 'fighting', 'assault',
            'beat', 'stab', 'suicide', 'harassment', 'harass', 'abuse',
            
            # ==================== CONTENIDO DISCRIMINATORIO/ODIO ====================
            # Español
            'racista', 'racismo', 'xenófobo', 'xenofobia', 'homofóbico', 'homofobia',
            'transfóbico', 'transfobia', 'machista', 'machismo', 'sexista', 'sexismo',
            'nazi', 'hitler', 'fascista', 'fascismo', 'supremacista', 'kkk',
            'odio', 'discriminar', 'discriminación', 'insulto', 'insultar', 'ofender',
            'amenaza', 'amenazar', 'acoso', 'acosar', 'bullying',
            
            # Inglés
            'racist', 'racism', 'xenophobic', 'xenophobia', 'homophobic', 'homophobia',
            'transphobic', 'transphobia', 'sexist', 'sexism', 'nazi', 'hitler',
            'fascist', 'fascism', 'supremacist', 'kkk', 'hate', 'discriminate',
            'discrimination', 'insult', 'offend', 'threat', 'threaten', 'harassment',
            'bullying',
            
            # ==================== CONTENIDO ILEGAL/PELIGROSO ====================
            # Español
            'droga', 'drogas', 'marihuana', 'cocaína', 'heroína', 'lsd', 'éxtasis',
            'alcohol', 'borracho', 'embriagado', 'drogado', 'traficante', 'tráfico',
            'pedofilia', 'pedófilo', 'zoofilia', 'necrofilia',
            'estafa', 'estafar', 'robo', 'robar', 'hurto', 'hurtar', 'fraude',
            'piratería', 'hackear', 'virus', 'malware', 'peligro',
            
            # Inglés
            'drug', 'drugs', 'marijuana', 'cocaine', 'heroin', 'lsd', 'ecstasy',
            'alcohol', 'drunk', 'intoxicated', 'drugged', 'dealer', 'trafficking',
            'pedophilia', 'pedophile', 'zoophilia', 'necrophilia',
            'scam', 'fraud', 'robbery', 'steal', 'theft', 'piracy', 'hack', 'virus',
            'malware', 'danger',
            
            # ==================== CONTENIDO NO APTO PARA EL SISTEMA ====================
            # Español
            'spam', 'publicidad', 'anuncio', 'comercial', 'venta', 'comprar', 'vender',
            'estafa', 'phishing', 'encuesta', 'sorteo', 'premio', 'gratis', 'oferta',
            'marketing', 'promoción', 'clickbait', 'sensacionalista', 'falso', 'falsa',
            'trampa', 'engaño', 'mentira', 'farsa',
            
            # Inglés
            'spam', 'advertisement', 'ad', 'commercial', 'sale', 'buy', 'sell',
            'scam', 'phishing', 'survey', 'giveaway', 'prize', 'free', 'offer',
            'marketing', 'promotion', 'clickbait', 'sensationalist', 'fake', 'false',
            'trick', 'deception', 'lie', 'hoax'
        }
        return set(word.lower() for word in banned_words)
    
    def validate_text_content(self, text: str, is_filename: bool = False) -> Tuple[bool, List[str]]:
        """
        Validar texto contra lista negra
        
        Args:
            text: Texto a validar
            is_filename: Si es nombre de archivo, buscar subcadenas
        """
        if not text:
            return True, []
        
        text_lower = text.lower()
        found_banned_words = []
        
        for word in self.banned_words:
            if is_filename:
                # PARA NOMBRES DE ARCHIVO: Buscar subcadenas
                if word in text_lower:
                    found_banned_words.append(word)
            else:
                # PARA DESCRIPCIONES: Buscar palabras completas
                if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
                    found_banned_words.append(word)
        
        is_valid = len(found_banned_words) == 0
        return is_valid, found_banned_words
    
    def validate_image_metadata(self, filename: str, description: str = None) -> Tuple[bool, List[str]]:
        """Validar nombre de archivo y descripción"""
        violations = []
        
        # Validar nombre de archivo (BUSCAR SUBCADENAS)
        filename_valid, filename_banned = self.validate_text_content(filename, is_filename=True)
        if not filename_valid:
            violations.extend(filename_banned)
        
        # Validar descripción (BUSCAR PALABRAS COMPLETAS)
        if description:
            desc_valid, desc_banned = self.validate_text_content(description, is_filename=False)
            if not desc_valid:
                violations.extend(desc_banned)
        
        return len(violations) == 0, violations


# Instancia singleton
content_validator = ContentValidationService()