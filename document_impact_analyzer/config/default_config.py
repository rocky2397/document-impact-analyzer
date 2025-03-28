# Default configuration for document impact analysis

# Default topic configuration
DEFAULT_TOPIC = "protectionism"

# Topic-specific keywords dictionaries
TOPICS = {
    "protectionism": {
        "name": "Protectionism",
        "description": "Analysis of trade barriers and protectionist policies",
        "keywords": [
            "protectionism", "trade barrier", "tariff", "import duty", "quota", 
            "trade war", "trade conflict", "trade tension", "trade restriction",
            "domestic industry protection", "import substitution", "local content requirement",
            "export subsidy", "anti-dumping", "countervailing duty", "safeguard measure",
            "economic nationalism", "national security", "border tax", "customs duty",
            "trade deficit", "political risk", "sanction", "embargo", "foreign investment restriction",
            "nationalization", "expropriation", "deglobalization", "trade policy"
        ]
    },
    "climate_change": {
        "name": "Climate Change",
        "description": "Analysis of climate change impacts and policies",
        "keywords": [
            "climate change", "global warming", "greenhouse gas", "carbon emissions", "CO2",
            "paris agreement", "renewable energy", "sustainability", "fossil fuels",
            "climate risk", "carbon tax", "carbon footprint", "net zero", "climate policy",
            "emission reduction", "extreme weather", "climate adaptation", "climate mitigation",
            "clean energy", "carbon neutral", "energy transition", "climate action",
            "IPCC", "sea level rise", "decarbonization", "climate crisis"
        ]
    },
    "digital_transformation": {
        "name": "Digital Transformation",
        "description": "Analysis of digital transformation impacts and strategies",
        "keywords": [
            "digital transformation", "digitalization", "automation", "artificial intelligence", "AI",
            "machine learning", "big data", "cloud computing", "digital strategy", "IoT",
            "internet of things", "industry 4.0", "digital disruption", "digital innovation",
            "digital technology", "smart technology", "digital solutions", "digital ecosystem",
            "digital platform", "digital infrastructure", "digital skills", "digital workforce",
            "digital adoption", "data analytics", "digital business model", "digital economy"
        ]
    }
}

# Impact scoring configuration
IMPACT_SCORING = {
    "relevance_weight": 0.6,
    "sentiment_weight": 0.4,
    "mention_weight": 0.2,
    "low_risk_threshold": 0.5,
    "high_risk_threshold": 1.5
}

# Visualization settings
VISUALIZATION = {
    "color_map": {
        "Low Risk": "green",
        "Moderate Risk": "orange",
        "High Risk": "red"
    }
}

# Recommendation templates
RECOMMENDATIONS = {
    "high_risk": [
        "Diversify supply chains to reduce dependency on specific regions",
        "Develop contingency plans for potential disruptions",
        "Increase focus on local operations in key markets",
        "Monitor regulatory changes closely in all operating regions"
    ],
    "moderate_risk": [
        "Review dependencies in high-risk regions",
        "Develop mitigation strategies for potential barriers",
        "Engage with industry groups on policy advocacy"
    ],
    "low_risk": [
        "Continue monitoring policy developments",
        "Maintain existing risk management practices"
    ]
}