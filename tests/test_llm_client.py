"""Unit tests for ai/llm_client.py"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

from ai.llm_client import LLMClient, LLMAnalysisResult


class TestLLMClient:
    """Tests for LLMClient class."""

    def test_client_initialization_without_api_key(self):
        """Test client initialization when API key is not configured."""
        with patch('ai.llm_client.VSE_LLM_API_KEY', None):
            client = LLMClient(api_key=None)
            assert client.client is None
            assert client.model is not None

    def test_client_initialization_with_api_key(self):
        """Test client initialization with API key."""
        with patch('ai.llm_client.VSE_LLM_API_KEY', 'test-key'):
            with patch('ai.llm_client.VSE_LLM_API_BASE', 'https://api.test.com/v1'):
                with patch('ai.llm_client.LLM_MODEL', 'test-model'):
                    client = LLMClient()
                    assert client.client is not None
                    assert client.api_key == 'test-key'
                    assert client.model == 'test-model'

    def test_extract_json_valid_response(self):
        """Test JSON extraction from valid LLM response."""
        with patch('ai.llm_client.VSE_LLM_API_KEY', None):
            client = LLMClient()
            
            json_text = '''{
                "signals": {
                    "BTCUSDT": {"signal": "buy", "confidence": 75, "reason": "test"}
                },
                "portfolio": {
                    "total_allocated_margin_usdt": 100,
                    "total_risk_usdt": 10
                }
            }'''
            
            result = client._extract_json(json_text)
            assert result is not None
            assert "signals" in result
            assert "portfolio" in result

    def test_extract_json_with_markdown(self):
        """Test JSON extraction from markdown-wrapped response."""
        with patch('ai.llm_client.VSE_LLM_API_KEY', None):
            client = LLMClient()
            
            json_text = '''```json
{
    "signals": {
        "ETHUSDT": {"signal": "sell", "confidence": 60, "reason": "test"}
    },
    "portfolio": {
        "total_allocated_margin_usdt": 50,
        "total_risk_usdt": 5
    }
}
```'''
            
            result = client._extract_json(json_text)
            assert result is not None
            assert "signals" in result

    def test_extract_json_invalid_response(self):
        """Test JSON extraction from invalid response."""
        with patch('ai.llm_client.VSE_LLM_API_KEY', None):
            client = LLMClient()
            
            result = client._extract_json("not json at all")
            assert result is None

    def test_extract_json_none_text(self):
        """Test JSON extraction from None text."""
        with patch('ai.llm_client.VSE_LLM_API_KEY', None):
            client = LLMClient()
            
            with pytest.raises(Exception) as exc_info:
                client._extract_json(None)
            assert "none" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_analyze_all_coins_no_client(self):
        """Test analyze_all_coins when client is None."""
        with patch('ai.llm_client.VSE_LLM_API_KEY', None):
            client = LLMClient()
            
            result = await client.analyze_all_coins({})
            assert result == {}

    @pytest.mark.asyncio
    async def test_get_all_signals_no_client(self):
        """Test get_all_signals when client is None."""
        with patch('ai.llm_client.VSE_LLM_API_KEY', None):
            client = LLMClient()
            
            result = await client.get_all_signals([])
            assert result == {}

    @pytest.mark.asyncio
    async def test_get_all_signals_converts_list_to_dict(self):
        """Test get_all_signals converts list to dict correctly."""
        with patch('ai.llm_client.VSE_LLM_API_KEY', 'test-key'):
            with patch('ai.llm_client.VSE_LLM_API_BASE', 'https://api.test.com/v1'):
                with patch('ai.llm_client.LLM_MODEL', 'test-model'):
                    client = LLMClient()
                    
                    coins_data = [
                        {"coin": "BTCUSDT", "price": 50000, "rsi": 45},
                        {"coin": "ETHUSDT", "price": 3000, "rsi": 55},
                    ]
                    
                    # Mock analyze_all_coins to return signals
                    with patch.object(client, 'analyze_all_coins', new_callable=AsyncMock) as mock_analyze:
                        from ai.ensemble_voter import LLMSignal
                        
                        mock_analyze.return_value = {
                            "BTCUSDT": LLMSignal(
                                signal="BUY",
                                confidence=0.7,
                                reasoning="Test",
                                model="test",
                            )
                        }
                        
                        result = await client.get_all_signals(coins_data)
                        assert "BTCUSDT" in result


class TestLLMAnalysisResult:
    """Tests for LLMAnalysisResult dataclass."""

    def test_creation(self):
        """Test creating LLMAnalysisResult instance."""
        result = LLMAnalysisResult(
            signal="BULLISH",
            confidence=0.75,
            reasoning="RSI oversold",
            model="test-model",
            timestamp="2024-01-01T00:00:00",
        )
        
        assert result.signal == "BULLISH"
        assert result.confidence == 0.75
        assert result.reasoning == "RSI oversold"
        assert result.model == "test-model"
        assert result.timestamp == "2024-01-01T00:00:00"


class TestLLMSignalIntegration:
    """Integration tests for LLMClient with ensemble voter."""

    def test_llm_signal_compatible_with_ensemble_voter(self):
        """Test that LLMClient output is compatible with EnsembleVoter."""
        from ai.ensemble_voter import LLMSignal
        
        # Create signal like LLMClient would
        signal = LLMSignal(
            signal="BUY",
            confidence=0.8,
            reasoning="Strong bullish signal",
            model="deepseek/deepseek-v3.2-speciale",
        )
        
        # Verify it has all required attributes
        assert hasattr(signal, 'signal')
        assert hasattr(signal, 'confidence')
        assert hasattr(signal, 'reasoning')
        assert hasattr(signal, 'model')
        
        # Verify values
        assert signal.signal == "BUY"
        assert signal.confidence == 0.8
        assert signal.reasoning == "Strong bullish signal"
        assert "deepseek" in signal.model


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
