import pandas as pd
import numpy as np
import networkx as nx
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os
import joblib

class InsiderThreatModel:
    def __init__(self, contamination=0.03, model_path="model/insider_iforest.pkl"):
        self.model_path = model_path
        # Smaller model for fast training
        self.model = IsolationForest(
            n_estimators=50,
            max_samples=1000,
            contamination=contamination,
            n_jobs=-1,
            random_state=42
        )
        self.features_df = None
        self.raw_logs = {}

        # Try to load existing saved model
        if os.path.exists(self.model_path):
            try:
                loaded = joblib.load(self.model_path)
                self.__dict__.update(loaded.__dict__)
                print(f"✅ Loaded existing model from '{self.model_path}'")
            except Exception as e:
                print(f"❌ Failed to load model: {e}")
        else:
            print(f"⚠️ No saved model found at '{self.model_path}'. Train before using predictions.")

    def _read_csv(self, path, nrows=None):
        """Read CSV and normalize timestamp column"""
        if not os.path.exists(path):
            return pd.DataFrame()
        
        df = pd.read_csv(path, nrows=nrows)
        
        # Look for timestamp/date/time columns
        timestamp_col = None
        for col in ["timestamp", "date", "time", "sent_time"]:
            if col in df.columns:
                timestamp_col = col
                break
        
        if timestamp_col:
            df['timestamp'] = pd.to_datetime(df[timestamp_col], errors='coerce')
        else:
            # If no timestamp/date column exists, create a dummy timestamp
            df['timestamp'] = pd.Timestamp('2000-01-01')  # arbitrary default
        
        # Ensure a date column exists for daily aggregation
        df['date'] = df['timestamp'].dt.date
        
        return df


    def load_data(self, data_folder="data", max_rows=50000):
        # Load raw CSVs (downsample for speed)
        self.logon = self._read_csv(f"{data_folder}/logon.csv", nrows=max_rows)
        self.device = self._read_csv(f"{data_folder}/device.csv", nrows=max_rows)
        self.email = self._read_csv(f"{data_folder}/email.csv", nrows=max_rows)
        self.file = self._read_csv(f"{data_folder}/file.csv", nrows=max_rows)

        # Normalize "user" column naming
        for df in [self.logon, self.device, self.email, self.file]:
            for alt in ["employee", "user_id", "actor"]:
                if "user" not in df.columns and alt in df.columns:
                    df.rename(columns={alt: "user"}, inplace=True)

        # Normalize email recipient columns
        if "recipient" not in self.email.columns:
            for col in ["to", "cc", "bcc"]:
                if col in self.email.columns:
                    self.email.rename(columns={col: "recipient"}, inplace=True)
                    break
        if "recipient" not in self.email.columns:
            self.email["recipient"] = "unknown"

        self.raw_logs = {"logon": self.logon, "device": self.device, "email": self.email, "file": self.file}

        # ----------- Temporal Features -----------
        if not self.logon.empty and "timestamp" in self.logon.columns:
            self.logon["hour"] = self.logon["timestamp"].dt.hour
            login_feats = self.logon.groupby("user").agg(
                mean_login_hour=("hour", "mean"),
                mean_logout_hour=("hour", "max")
            ).reset_index()
        else:
            login_feats = pd.DataFrame(columns=["user", "mean_login_hour", "mean_logout_hour"])

        # ----------- Usage Counts -----------
        file_feats = self.file.groupby("user").agg(files_per_day=("timestamp", "count")).reset_index() if not self.file.empty else pd.DataFrame(columns=["user", "files_per_day"])
        usb_feats = self.device.groupby("user").agg(usb_per_day=("timestamp", "count")).reset_index() if not self.device.empty else pd.DataFrame(columns=["user", "usb_per_day"])
        email_feats = self.email.groupby("user").agg(emails_per_day=("timestamp", "count")).reset_index() if not self.email.empty else pd.DataFrame(columns=["user", "emails_per_day"])

        # ----------- Out-of-session (after-hours) -----------
        if not self.logon.empty and "hour" in self.logon.columns:
            self.logon["after_hours"] = self.logon["hour"].isin(list(range(0,6)) + list(range(20,24))).astype(int)
            out_session = self.logon.groupby("user").agg(out_of_session_access=("after_hours","sum")).reset_index()
        else:
            out_session = pd.DataFrame(columns=["user", "out_of_session_access"])

        # ----------- Graph Features (degree centrality only) -----------
        if not self.email.empty:
            G = nx.from_pandas_edgelist(self.email, "user", "recipient", create_using=nx.Graph())
            centrality = pd.DataFrame({
                "user": list(G.nodes),
                "degree_centrality": pd.Series(nx.degree_centrality(G)),
                "betweenness_centrality": 0  # skip betweenness for speed
            })
        else:
            centrality = pd.DataFrame(columns=["user", "degree_centrality", "betweenness_centrality"])

        # ----------- Email Text Features -----------
        if "subject" in self.email.columns:
            self.email["subject_len"] = self.email["subject"].astype(str).str.len()
            self.email["keyword_flag"] = self.email["subject"].str.contains(
                "confidential|secret|password", case=False, na=False
            ).astype(int)
            self.email["sentiment"] = np.random.choice([0, 1], size=len(self.email))  # placeholder
            text_feats = self.email.groupby("user").agg(
                subject_len=("subject_len", "mean"),
                keyword_flag=("keyword_flag", "mean"),
                sentiment=("sentiment", "mean")
            ).reset_index()
        else:
            text_feats = pd.DataFrame({"user": self.email["user"].unique() if not self.email.empty else [],
                                       "subject_len": 0,
                                       "keyword_flag": 0,
                                       "sentiment": 0})

        # ----------- Merge All Features -----------
        features = (login_feats.merge(file_feats, on="user", how="outer")
                    .merge(usb_feats, on="user", how="outer")
                    .merge(email_feats, on="user", how="outer")
                    .merge(out_session, on="user", how="outer")
                    .merge(centrality, on="user", how="outer")
                    .merge(text_feats, on="user", how="outer")
                    .fillna(0))

        self.features_df = features
        return features

    def train(self, save=True):
        if self.features_df is None:
            raise ValueError("No features. Run load_data() first.")

        X = self.features_df.drop(columns=["user"]).astype(float).fillna(0)

        # Train IsolationForest
        self.model.fit(X)

        # Compute anomaly scores
        if_scores = -self.model.decision_function(X)

        # Normalize scores
        scaler = StandardScaler()
        self.features_df["isolation_forest"] = scaler.fit_transform(if_scores.reshape(-1,1)).flatten()

        # Rank users
        self.features_df["rank"] = self.features_df["isolation_forest"].rank(ascending=False).astype(int)

        if save:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self, self.model_path)
            print(f"✅ Model trained and saved to {self.model_path}")

        return self.features_df

    def get_risky_users(self, top_n=20):
        if self.features_df is None:
            return []
        ranked = self.features_df.sort_values("isolation_forest", ascending=False)
        return ranked.head(top_n).to_dict(orient="records")
    # Inside InsiderThreatModel class

    def get_user_features(self, user):
        """Return per-day aggregated features for a given user safely."""
        if not self.raw_logs:
            return []

        # Helper to compute per-day counts safely
        def daily_count(df, col_name):
            if df is None or df.empty or 'user' not in df.columns or 'date' not in df.columns:
                return pd.DataFrame(columns=['date', col_name])
            user_df = df[df['user'] == user]
            if user_df.empty:
                return pd.DataFrame(columns=['date', col_name])
            counts = user_df.groupby('date')['user'].count().reset_index()
            counts = counts.rename(columns={'user': col_name})
            return counts

        logon_counts = daily_count(self.logon, 'logons')
        file_counts = daily_count(self.file, 'files')
        usb_counts = daily_count(self.device, 'usb')
        email_counts = daily_count(self.email, 'emails')

        # Merge all per-day counts
        from functools import reduce
        dfs = [logon_counts, file_counts, usb_counts, email_counts]
        merged = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer'), dfs)
        if merged.empty:
            merged = pd.DataFrame({'date': [], 'logons': [], 'files': [], 'usb': [], 'emails': []})

        merged = merged.fillna(0)

        # Add mean risk if available
        if self.features_df is not None and 'isolation_forest' in self.features_df.columns:
            user_row = self.features_df[self.features_df['user'] == user]
            merged['mean_risk'] = float(user_row['isolation_forest'].values[0]) if not user_row.empty else 0
        else:
            merged['mean_risk'] = 0

        merged['user'] = user
        merged['date'] = merged['date'].astype(str)  # ensure JSON serializable

        return merged.to_dict(orient='records')



    
    def get_user_raw(self, user):
        """Return raw log data per user, organized by source."""
        if not self.raw_logs:
            return {}
        
        user_raw = {}
        for src, df in self.raw_logs.items():
            if df.empty or 'user' not in df.columns:
                user_raw[src] = []
            else:
                # Filter only this user and convert timestamp to string for JSON
                temp = df[df['user'] == user].copy()
                if 'timestamp' in temp.columns:
                    temp['timestamp'] = temp['timestamp'].astype(str)
                user_raw[src] = temp.to_dict(orient='records')
        return user_raw
