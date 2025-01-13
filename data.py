import numpy as np
import os.path as osp
from tqdm.auto import tqdm
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import warnings
from torchvision.transforms import (
    RandomResizedCrop, 
    CenterCrop, 
    RandomEqualize, 
    RandomPerspective, 
    Compose,
    Normalize, 
    ToTensor, 
    RandomHorizontalFlip, 
    Resize, 
    ColorJitter
)
from sklearn.model_selection import train_test_split
from datetime import datetime
from scipy.stats import norm
import cv2
from typing import Dict, Tuple, List
from geopy.distance import geodesic
from transformers import AutoFeatureExtractor

tqdm.pandas()

class RegressionDataset(Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.transform = transform
        self.df = df

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = row['image_path']
        score = row['trueskill_score']
        
        img = Image.open(name)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img[:,:,0] = clahe.apply(img[:,:,0])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        return {
            "pixel_values": img,
            "label": score
        }

    def __len__(self):
        return len(self.df)

def compute_temporal_decay(timestamp, half_life_years=4.0):
    """Temporal decay factor λ(t) = exp(-ln(2)t/Thalf)"""
    current_time = datetime.now()
    time_diff_years = (current_time - timestamp).days / 365.0
    return np.exp(-np.log(2) * time_diff_years / half_life_years)

def compute_spatial_influence(loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
    """Spatial influence factor f(d) = 1 - exp(-3d)"""
    distance = geodesic(loc1, loc2).kilometers
    return 1 - np.exp(-3 * distance)

def v_win_function(t, eps):
    """VI(>ε)(t,ε) = N(t-ε)/Φ(t-ε)"""
    denom = norm.cdf(t - eps)
    if denom < 1e-10:
        return 0
    return norm.pdf(t - eps) / denom

def w_win_function(t, eps):
    """WI(>ε)(t,ε) = v_win(t,ε) * (v_win(t,ε) + t - ε)"""
    v = v_win_function(t, eps)
    return v * (v + t - eps)

def v_draw_function(t, eps):
    """VI(|.|≤ε)(t,ε) = (N(-ε-t) - N(ε-t))/(Φ(ε-t) - Φ(-ε-t))"""
    numerator = norm.pdf(-eps - t) - norm.pdf(eps - t)
    denominator = norm.cdf(eps - t) - norm.cdf(-eps - t)
    if abs(denominator) < 1e-10:
        return 0
    return numerator / denominator

def w_draw_function(t, eps):
    """WI(|.|≤ε)(t,ε) = v_draw²(t,ε) + ((ε-t)N(ε-t) + (ε+t)N(-ε-t))/(Φ(ε-t) - Φ(-ε-t))"""
    v = v_draw_function(t, eps)
    numerator = ((eps-t)*norm.pdf(eps-t) + (eps+t)*norm.pdf(-eps-t))
    denominator = norm.cdf(eps-t) - norm.cdf(-eps-t)
    if abs(denominator) < 1e-10:
        return 0
    return v**2 + numerator/denominator

def update_trueskill_scores(winner_score, loser_score, t_decay, s_influence, 
                          beta=4.317, draw_margin=0.1, is_draw=False):
    """Update TrueSkill scores using the complete formulas from the paper"""
    mu_winner, sigma_winner = winner_score
    mu_loser, sigma_loser = loser_score
    
    # Calculate c²
    c_squared = 2 * (beta ** 2) + sigma_winner ** 2 + sigma_loser ** 2
    c = np.sqrt(c_squared)
    
    # Performance difference
    perf_diff = (mu_winner - mu_loser) / c
    eps_scaled = draw_margin / c
    
    # Calculate v and w based on game outcome
    if is_draw:
        v = v_draw_function(perf_diff, eps_scaled)
        w = w_draw_function(perf_diff, eps_scaled)
    else:
        v = v_win_function(perf_diff, eps_scaled)
        w = w_win_function(perf_diff, eps_scaled)
    
    # Update μ values
    mu_winner_new = mu_winner + t_decay * (sigma_winner**2/c) * v * s_influence
    mu_loser_new = mu_loser - t_decay * (sigma_loser**2/c) * v * s_influence
    
    # Update σ² values
    sigma_winner_new = sigma_winner * np.sqrt(1 - t_decay * (sigma_winner**2/c_squared) * w * s_influence)
    sigma_loser_new = sigma_loser * np.sqrt(1 - t_decay * (sigma_loser**2/c_squared) * w * s_influence)
    
    return (mu_winner_new, sigma_winner_new), (mu_loser_new, sigma_loser_new)

def preprocess_csv(args, df):
    """Process comparison data using TrueSkill algorithm"""
    # Initialize parameters
    scores = {}  
    initial_mu = 25.0
    initial_sigma = 8.333
    beta = 4.317
    draw_margin = 0.1
    
    # Convert timestamps
    df['vote_timestamp'] = pd.to_datetime(df['vote_timestamp'])
    
    # Process each comparison
    for idx, row in tqdm(df.iterrows(), desc="Computing TrueSkill scores"):
        left_path = row['left_image_path']
        right_path = row['right_image_path']
        
        # Initialize scores if needed
        if left_path not in scores:
            scores[left_path] = (initial_mu, initial_sigma)
        if right_path not in scores:
            scores[right_path] = (initial_mu, initial_sigma)
            
        # Get temporal decay and spatial influence
        t_decay = compute_temporal_decay(row['vote_timestamp'])
        s_influence = compute_spatial_influence(
            row['left_image_location'],
            row['right_image_location']
        )
        
        # Determine winner/loser
        if row['choice'] == 'left':
            winner_path, loser_path = left_path, right_path
        else:
            winner_path, loser_path = right_path, left_path
            
        # Update scores
        scores[winner_path], scores[loser_path] = update_trueskill_scores(
            scores[winner_path],
            scores[loser_path],
            t_decay,
            s_influence,
            beta=beta,
            draw_margin=draw_margin,
            is_draw=False
        )
    
    # Create final dataframe
    result_df = pd.DataFrame([
        {'image_path': path, 'trueskill_score': mu - 3*sigma}
        for path, (mu, sigma) in scores.items()
    ])
    
    # Apply classification thresholds
    mu = result_df['trueskill_score'].mean()
    std = result_df['trueskill_score'].std()
    
    def assign_class(score):
        if score <= mu - std:
            return 0  # Impoverished
        elif score >= mu + std:
            return 2  # Affluent
        return 1  # Middle
    
    result_df['label'] = result_df['trueskill_score'].apply(assign_class)
    
    if args.num_classes == 2:
        result_df = result_df[result_df['label'].isin([0, 2])]
        result_df['label'] = result_df['label'].apply(lambda x: 0 if x == 0 else 1)
    
    return result_df

def build_transform(args, image_mean, image_std, size=384):
    normalize = Normalize(mean=image_mean, std=image_std)
    
    train_transform = Compose([
        Resize((512, 512)),
        CenterCrop((384, 384)),
        RandomPerspective(),
        ToTensor(),
        normalize,
    ])

    test_transform = val_transform = Compose([
        Resize((384, 384)),
        ToTensor(),
        normalize,
    ])
    
    return train_transform, val_transform, test_transform

def build_dataset_and_feacture(args):
    root = args.root
    csv_path = args.csv_path
    df = pd.read_csv(csv_path)
    df = preprocess_csv(args, df)
    
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
    
    train_transform, val_transform, test_transform = build_transform(
        args,
        feature_extractor.image_mean,
        feature_extractor.image_std,
        feature_extractor.size
    )
    
    df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_set = RegressionDataset(train_df, train_transform)
    val_set = RegressionDataset(val_df, val_transform)
    test_set = RegressionDataset(test_df, test_transform)

    return (train_set, val_set, test_set), feature_extractor