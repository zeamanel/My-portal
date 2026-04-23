def apply_musa_context(user_prompt: str) -> str:
    """
    Wraps the user's prompt with the Musa Context:
    High-fashion streetwear in Addis Ababa, featuring traditional Habesha Tilf embroidery patterns 
    integrated into modern techwear, set against the backdrop of the Meskel Square skyline, 
    8k resolution, cinematic lighting.
    """
    musa_context = (
        "High-fashion streetwear in Addis Ababa, featuring traditional Habesha Tilf embroidery patterns "
        "integrated into modern techwear, set against the backdrop of the Meskel Square skyline, "
        "8k resolution, cinematic lighting."
    )
    
    # Check if context is already present to prevent duplication if called multiple times (optional safety)
    if musa_context in user_prompt:
        return user_prompt
        
    return f"{user_prompt}, {musa_context}"
