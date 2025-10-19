from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from scipy import stats
import math
import random
from datetime import datetime
import json

app = Flask(__name__)

class FootballPredictor:
    def __init__(self):
        self.markets = [
            "1X2", "Chance Dupla", "Ambas A Marcar (GG/NG)", 
            "Total De Golos", "Booking 1X2", "Empate Anula Aposta", 
            "Primeira Equipa A Marcar", "Total De Golos Ímpar/Par", 
            "1ª Parte - 1X2", "Handicap", "Vencedor e total de golos", 
            "Total de cartões", "Qual equipe vencera o resto da partida"
        ]
        
    def calculate_implied_probability(self, odds):
        """Convert decimal odds to implied probability"""
        if odds <= 1:
            return 0
        return 100 / odds
    
    def normalize_probabilities(self, home_prob, draw_prob, away_prob):
        """Normalize probabilities to sum to 100%"""
        total = home_prob + draw_prob + away_prob
        if total == 0:
            return 0, 0, 0
        return (home_prob/total)*100, (draw_prob/total)*100, (away_prob/total)*100
    
    def calculate_edge(self, model_prob, market_prob):
        """Calculate the edge between model and market probabilities"""
        return model_prob - market_prob
    
    def calculate_confidence_score(self, model_prob, market_prob, edge):
        """Calculate a confidence score for a betting suggestion"""
        # Base score on edge and model probability
        edge_weight = 1.5
        confidence_weight = 0.8
        
        score = (edge_weight * edge) + (confidence_weight * (model_prob - 50))
        
        # Normalize to 0-100 scale
        return max(0, min(100, 50 + score))
    
    def analyze_1x2(self, home_prob, draw_prob, away_prob, home_odds, draw_odds, away_odds):
        """Analyze 1X2 market"""
        # Get implied probabilities from odds
        home_implied = self.calculate_implied_probability(home_odds)
        draw_implied = self.calculate_implied_probability(draw_odds)
        away_implied = self.calculate_implied_probability(away_odds)
        
        # Normalize implied probabilities
        home_implied, draw_implied, away_implied = self.normalize_probabilities(
            home_implied, draw_implied, away_implied
        )
        
        results = []
        outcomes = [
            ("Casa", home_prob, home_implied, home_odds),
            ("Empate", draw_prob, draw_implied, draw_odds),
            ("Fora", away_prob, away_implied, away_odds)
        ]
        
        for outcome, model_p, implied_p, odds in outcomes:
            edge = self.calculate_edge(model_p, implied_p)
            confidence = self.calculate_confidence_score(model_p, implied_p, edge)
            
            results.append({
                "market": "1X2",
                "selection": outcome,
                "model_prob": round(model_p, 1),
                "implied_prob": round(implied_p, 1),
                "edge": round(edge, 1),
                "odds": odds,
                "confidence": round(confidence, 1)
            })
        
        # Sort by confidence (descending)
        return sorted(results, key=lambda x: x["confidence"], reverse=True)
    
    def analyze_gg_ng(self, both_score_prob, gg_odds=1.85, ng_odds=1.95):
        """Analyze Both Teams to Score market"""
        gg_implied = self.calculate_implied_probability(gg_odds)
        ng_implied = self.calculate_implied_probability(ng_odds)
        
        # Normalize
        total = gg_implied + ng_implied
        gg_implied = (gg_implied / total) * 100
        ng_implied = (ng_implied / total) * 100
        
        ng_prob = 100 - both_score_prob
        
        results = []
        
        # GG analysis
        gg_edge = self.calculate_edge(both_score_prob, gg_implied)
        gg_confidence = self.calculate_confidence_score(both_score_prob, gg_implied, gg_edge)
        
        results.append({
            "market": "Ambas A Marcar (GG/NG)",
            "selection": "Sim",
            "model_prob": round(both_score_prob, 1),
            "implied_prob": round(gg_implied, 1),
            "edge": round(gg_edge, 1),
            "odds": gg_odds,
            "confidence": round(gg_confidence, 1)
        })
        
        # NG analysis
        ng_edge = self.calculate_edge(ng_prob, ng_implied)
        ng_confidence = self.calculate_confidence_score(ng_prob, ng_implied, ng_edge)
        
        results.append({
            "market": "Ambas A Marcar (GG/NG)",
            "selection": "Não",
            "model_prob": round(ng_prob, 1),
            "implied_prob": round(ng_implied, 1),
            "edge": round(ng_edge, 1),
            "odds": ng_odds,
            "confidence": round(ng_confidence, 1)
        })
        
        return sorted(results, key=lambda x: x["confidence"], reverse=True)
    
    def analyze_over_under(self, over_prob, over_odds=2.08, under_odds=1.77):
        """Analyze Over/Under 2.5 goals market"""
        over_implied = self.calculate_implied_probability(over_odds)
        under_implied = self.calculate_implied_probability(under_odds)
        
        # Normalize
        total = over_implied + under_implied
        over_implied = (over_implied / total) * 100
        under_implied = (under_implied / total) * 100
        
        under_prob = 100 - over_prob
        
        results = []
        
        # Over analysis
        over_edge = self.calculate_edge(over_prob, over_implied)
        over_confidence = self.calculate_confidence_score(over_prob, over_implied, over_edge)
        
        results.append({
            "market": "Total De Golos",
            "selection": "Over 2.5",
            "model_prob": round(over_prob, 1),
            "implied_prob": round(over_implied, 1),
            "edge": round(over_edge, 1),
            "odds": over_odds,
            "confidence": round(over_confidence, 1)
        })
        
        # Under analysis
        under_edge = self.calculate_edge(under_prob, under_implied)
        under_confidence = self.calculate_confidence_score(under_prob, under_implied, under_edge)
        
        results.append({
            "market": "Total De Golos",
            "selection": "Under 2.5",
            "model_prob": round(under_prob, 1),
            "implied_prob": round(under_implied, 1),
            "edge": round(under_edge, 1),
            "odds": under_odds,
            "confidence": round(under_confidence, 1)
        })
        
        return sorted(results, key=lambda x: x["confidence"], reverse=True)
    
    def analyze_first_goal(self, home_first_prob, home_odds=2.38, away_odds=1.57):
        """Analyze First Team to Score market"""
        home_implied = self.calculate_implied_probability(home_odds)
        away_implied = self.calculate_implied_probability(away_odds)
        
        # Normalize
        total = home_implied + away_implied
        home_implied = (home_implied / total) * 100
        away_implied = (away_implied / total) * 100
        
        away_first_prob = 100 - home_first_prob
        
        results = []
        
        # Home to score first
        home_edge = self.calculate_edge(home_first_prob, home_implied)
        home_confidence = self.calculate_confidence_score(home_first_prob, home_implied, home_edge)
        
        results.append({
            "market": "Primeira Equipa A Marcar",
            "selection": "Casa",
            "model_prob": round(home_first_prob, 1),
            "implied_prob": round(home_implied, 1),
            "edge": round(home_edge, 1),
            "odds": home_odds,
            "confidence": round(home_confidence, 1)
        })
        
        # Away to score first
        away_edge = self.calculate_edge(away_first_prob, away_implied)
        away_confidence = self.calculate_confidence_score(away_first_prob, away_implied, away_edge)
        
        results.append({
            "market": "Primeira Equipa A Marcar",
            "selection": "Fora",
            "model_prob": round(away_first_prob, 1),
            "implied_prob": round(away_implied, 1),
            "edge": round(away_edge, 1),
            "odds": away_odds,
            "confidence": round(away_confidence, 1)
        })
        
        return sorted(results, key=lambda x: x["confidence"], reverse=True)
    
    def analyze_handicap(self, home_cover_prob, home_odds=1.91, away_odds=1.91):
        """Analyze Handicap market (assuming -1 handicap for home team)"""
        home_implied = self.calculate_implied_probability(home_odds)
        away_implied = self.calculate_implied_probability(away_odds)
        
        # Normalize
        total = home_implied + away_implied
        home_implied = (home_implied / total) * 100
        away_implied = (away_implied / total) * 100
        
        away_cover_prob = 100 - home_cover_prob
        
        results = []
        
        # Home covers handicap
        home_edge = self.calculate_edge(home_cover_prob, home_implied)
        home_confidence = self.calculate_confidence_score(home_cover_prob, home_implied, home_edge)
        
        results.append({
            "market": "Handicap -1",
            "selection": "Casa",
            "model_prob": round(home_cover_prob, 1),
            "implied_prob": round(home_implied, 1),
            "edge": round(home_edge, 1),
            "odds": home_odds,
            "confidence": round(home_confidence, 1)
        })
        
        # Away covers handicap
        away_edge = self.calculate_edge(away_cover_prob, away_implied)
        away_confidence = self.calculate_confidence_score(away_cover_prob, away_implied, away_edge)
        
        results.append({
            "market": "Handicap -1",
            "selection": "Fora",
            "model_prob": round(away_cover_prob, 1),
            "implied_prob": round(away_implied, 1),
            "edge": round(away_edge, 1),
            "odds": away_odds,
            "confidence": round(away_confidence, 1)
        })
        
        return sorted(results, key=lambda x: x["confidence"], reverse=True)
    
    def get_recommendation_level(self, confidence):
        """Get recommendation level based on confidence score"""
        if confidence >= 70:
            return "APOSTAR FORTE", "high"
        elif confidence >= 50:
            return "CONSIDERAR APOSTA", "medium"
        else:
            return "EVITAR", "low"
    
    def generate_predictions(self, data):
        """Generate predictions for all markets based on input data"""
        home_team = data.get('home_team', 'Time A')
        away_team = data.get('away_team', 'Time B')
        
        home_prob = float(data.get('home_prob', 45))
        draw_prob = float(data.get('draw_prob', 30))
        away_prob = float(data.get('away_prob', 25))
        
        home_odds = float(data.get('home_odds', 2.2))
        draw_odds = float(data.get('draw_odds', 3.5))
        away_odds = float(data.get('away_odds', 3.1))
        
        both_score_prob = float(data.get('both_score_prob', 65))
        over_25_prob = float(data.get('over_25_prob', 55))
        first_goal_prob = float(data.get('first_goal_prob', 48))
        handicap_prob = float(data.get('handicap_prob', 40))
        
        # Run analyses for different markets
        predictions = []
        
        # 1X2 analysis
        predictions.extend(self.analyze_1x2(
            home_prob, draw_prob, away_prob, 
            home_odds, draw_odds, away_odds
        ))
        
        # GG/NG analysis
        predictions.extend(self.analyze_gg_ng(both_score_prob))
        
        # Over/Under analysis
        predictions.extend(self.analyze_over_under(over_25_prob))
        
        # First goal analysis
        predictions.extend(self.analyze_first_goal(first_goal_prob))
        
        # Handicap analysis
        predictions.extend(self.analyze_handicap(handicap_prob))
        
        # Add recommendation levels
        for pred in predictions:
            rec_text, rec_level = self.get_recommendation_level(pred['confidence'])
            pred['recommendation'] = rec_text
            pred['recommendation_level'] = rec_level
        
        # Sort by confidence (descending)
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Get top recommendation
        top_recommendation = predictions[0] if predictions else None
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'predictions': predictions,
            'top_recommendation': top_recommendation,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

# Initialize predictor
predictor = FootballPredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Generate predictions
        result = predictor.generate_predictions(data)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/markets')
def get_markets():
    return jsonify({'markets': predictor.markets})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
