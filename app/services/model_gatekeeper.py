# app/services/model_gatekeeper.py
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

class ModelGatekeeper:
    """Manages access to AI models based on database configuration"""
    
    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self._models_cache = {}
        self._cache_expiry = None
        self._cache_duration = 300  # 5 minutes
        
    async def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a single model by ID with validation
        """
        # Check cache first
        if model_id in self._models_cache and self._is_cache_valid():
            return self._models_cache.get(model_id)
        
        try:
            response = self.supabase.table("ai_models").select("*").eq(
                "model_id", model_id
            ).execute()
            
            if not response.data:
                return None
                
            model = response.data[0]
            
            # Validate model
            if not await self._validate_model(model):
                return None
            
            # Cache the model
            self._models_cache[model_id] = model
            return model
            
        except Exception as e:
            print(f"❌ Error fetching model {model_id}: {e}")
            return None
    
    async def list_available_models(
        self, 
        media_type: str = None,
        provider: str = None,
        max_price: int = None,
        active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        List available models with filtering options
        """
        try:
            query = self.supabase.table("ai_models").select("*")
            
            if active_only:
                query = query.eq("is_active", True).eq("maintenance_mode", False)
            
            if media_type:
                query = query.eq("media_type", media_type.title())
            
            if provider:
                query = query.eq("provider", provider)
            
            response = query.order("base_price_tokens").execute()
            
            if not response.data:
                return []
            
            # Filter by price if specified
            models = response.data
            if max_price is not None:
                models = [m for m in models if m.get('base_price_tokens', 0) <= max_price]
            
            # Validate each model
            valid_models = []
            for model in models:
                if await self._validate_model(model):
                    valid_models.append(model)
            
            return valid_models
            
        except Exception as e:
            print(f"❌ Error listing models: {e}")
            return []
    
    async def _validate_model(self, model: Dict[str, Any]) -> bool:
        """
        Validate if a model can be used
        """
        # Check if model is active
        if not model.get('is_active', False):
            return False
        
        # Check if in maintenance
        if model.get('maintenance_mode', False):
            return False
        
        # Check if has required API keys
        provider = model.get('provider')
        service_type = model.get('service_type')
        
        # Check if provider service is configured
        if not await self._check_provider_availability(provider, service_type):
            return False
        
        # Check if model has valid pricing
        if not self._validate_pricing(model):
            return False
        
        return True
    
    async def _check_provider_availability(self, provider: str, service_type: str) -> bool:
        """
        Check if the provider/service is configured and available
        """
        # Simple check - in production, you might want to ping the API
        required_keys = {
            'openrouter': 'OPENROUTER_API_KEY',
            'google': 'GEMINI_API_KEY',
            'fal': 'FAL_API_KEY',
            'suno': 'SUNO_API_KEY',
            'laozhang': 'LAOZHANG_API_KEY',
            'higgsfield': 'HIGGSFIELD_API_KEY'
        }
        
        if provider in required_keys:
            from app.core.config import Config
            key_name = required_keys[provider]
            return bool(getattr(Config, key_name, None))
        
        return True  # Assume other providers are available
    
    def _validate_pricing(self, model: Dict[str, Any]) -> bool:
        """
        Validate model pricing structure
        """
        # Must have base price
        if model.get('base_price_tokens') is None:
            return False
        
        # Quality prices should be non-negative
        quality_fields = ['price_1k', 'price_2k', 'price_4k', 'price_8k']
        for field in quality_fields:
            price = model.get(field)
            if price is not None and price < 0:
                return False
        
        return True
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        if not self._cache_expiry:
            return False
        
        return datetime.now() < self._cache_expiry
    
    def clear_cache(self):
        """Clear the model cache"""
        self._models_cache = {}
        self._cache_expiry = None
    
    async def calculate_model_cost(
        self, 
        model_id: str, 
        quality: str = "1K",
        duration: int = None,
        tokens: int = None,
        resolution: str = None
    ) -> Dict[str, Any]:
        """
        Calculate the cost for using a model with given parameters
        """
        model = await self.get_model(model_id)
        if not model:
            return {"error": f"Model {model_id} not available"}
        
        # Base cost
        base_cost = model.get('base_price_tokens', 0)
        
        # Quality premium
        quality_field = f"price_{quality.lower().replace('k', '')}k"
        quality_premium = model.get(quality_field, 0)
        
        # Additional cost calculations based on parameters
        additional_cost = 0
        
        if duration and model.get('base_cost_per_second'):
            additional_cost += duration * float(model['base_cost_per_second'])
        
        if tokens and model.get('base_cost_per_token'):
            additional_cost += tokens * float(model['base_cost_per_token'])
        
        if resolution and model.get('base_cost_per_pixel'):
            # Parse resolution like "1920x1080"
            try:
                width, height = map(int, resolution.split('x'))
                pixels = width * height
                additional_cost += pixels * float(model['base_cost_per_pixel'])
            except:
                pass
        
        total_cost = base_cost + quality_premium + additional_cost
        
        return {
            "model_id": model_id,
            "model_name": model.get('display_name'),
            "base_cost": base_cost,
            "quality_premium": quality_premium,
            "additional_costs": additional_cost,
            "total_cost": total_cost,
            "quality": quality,
            "provider": model.get('provider'),
            "service_type": model.get('service_type')
        }
    
    async def get_recommended_model(
        self,
        media_type: str,
        budget: int,
        quality: str = "1K",
        provider_preference: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best model recommendation based on requirements
        """
        available_models = await self.list_available_models(
            media_type=media_type,
            provider=provider_preference,
            active_only=True
        )
        
        if not available_models:
            return None
        
        # Filter by budget
        affordable_models = []
        for model in available_models:
            cost_calc = await self.calculate_model_cost(
                model['model_id'],
                quality=quality
            )
            
            if 'error' not in cost_calc and cost_calc['total_cost'] <= budget:
                model_with_cost = model.copy()
                model_with_cost['estimated_cost'] = cost_calc['total_cost']
                affordable_models.append(model_with_cost)
        
        if not affordable_models:
            return None
        
        # Sort by cost (cheapest first) then by provider preference
        affordable_models.sort(key=lambda x: (
            x['estimated_cost'],
            0 if x.get('provider') == provider_preference else 1
        ))
        
        return affordable_models[0]