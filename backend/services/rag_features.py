def rag_signal_to_features(signal: dict) -> dict:
    """Convert a signal dict to numeric RAG features."""
    sentiment_map = {
        "bullish": 1,
        "neutral": 0,
        "bearish": -1
    }

    sentiment_score = sentiment_map.get(signal.get("overall_sentiment", "neutral"), 0)

    return {
        "rag_sentiment": sentiment_score
            * signal.get("sentiment_strength", 0)
            * signal.get("confidence", 0),

        "rag_sentiment_strength": signal.get("sentiment_strength", 0),
        "rag_confidence": signal.get("confidence", 0),

        "num_bullish_drivers": len(signal.get("bullish_drivers", [])),
        "num_bearish_risks": len(signal.get("bearish_risks", [])),

        "event_present": int(len(signal.get("key_events", [])) > 0),
        "uncertainty_present": int(len(signal.get("uncertainty_flags", [])) > 0),
    }
