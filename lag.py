import pandas as pd
import re

class LagFeatureGenerator:
    def __init__(self, lot_col='LOT_ID', target_col='discharge',
                 n_lags=3, mark_missing_value=-999):
        self.lot_col = lot_col
        self.target_col = target_col
        self.n_lags = n_lags
        self.mark_missing_value = mark_missing_value
        self.group_stats_ = None

    def _parse_lot_info(self, lot_id):
        """
        LOT 네이밍에서 line, date, serial 번호 추출
        예: GMPCAN86L23041505 → line=L, date=230415, serial=05
        """
        match = re.match(r'.*([ILM])(\d{6})(\d{2})$', lot_id)
        if match:
            return match.groups()
        return (None, None, None)

    def _ensure_lot_info(self, df):
        """
        LOT 정보를 파싱해 LOT_group, LOT_line, LOT_date, LOT_serial 생성
        """
        lot_info = df[self.lot_col].apply(self._parse_lot_info)
        df['LOT_line'] = lot_info.apply(lambda x: x[0])
        df['LOT_date'] = lot_info.apply(lambda x: x[1])
        df['LOT_serial'] = lot_info.apply(lambda x: x[2])
        df['LOT_group'] = df['LOT_line'] + df['LOT_date']  # 그룹은 일자 기준
        return df

    def fit(self, df):
        df = df.copy()
        if self.lot_col not in df.columns:
            df[self.lot_col] = df.index  # LOT_ID가 index인 경우

        df = self._ensure_lot_info(df)
        df.sort_values(['LOT_line', 'LOT_date', 'LOT_serial'], inplace=True)

        self.group_stats_ = (
            df[['LOT_group', self.target_col]]
            .drop_duplicates(subset='LOT_group')
            .reset_index(drop=True)
        )
        return self

    def transform(self, df):
        df = df.copy()
        if self.lot_col not in df.columns:
            df[self.lot_col] = df.index  # LOT_ID가 index인 경우

        original_index = df.index.copy()

        df = self._ensure_lot_info(df)
        df = df.merge(
            self.group_stats_,
            on='LOT_group',
            how='left',
            suffixes=('', '_current')
        )

        self.group_stats_.sort_values(by='LOT_group', inplace=True)
        group_list = self.group_stats_['LOT_group'].tolist()
        discharge_list = self.group_stats_[self.target_col].tolist()

        lag_df = pd.DataFrame({'LOT_group': group_list})
        for i in range(1, self.n_lags + 1):
            lag_col = f'{self.target_col}_lag_{i}'
            lag_df[lag_col] = [self.mark_missing_value]*i + discharge_list[:-i]

        df = df.merge(lag_df, on='LOT_group', how='left')

        # ✅ 기존 컬럼 + lag feature만 남김
        lag_cols = [f'{self.target_col}_lag_{i}' for i in range(1, self.n_lags + 1)]
        result_df = df.drop(columns=[
            'LOT_group', 'LOT_line', 'LOT_date', 'LOT_serial', f'{self.target_col}_current'
        ], errors='ignore')

        # 순서 보장 위해 기존 컬럼 + lag 순서 지정
        existing_cols = [col for col in df.columns if col in result_df.columns and col not in lag_cols]
        result_df = result_df[existing_cols + lag_cols]

        result_df.index = original_index
        return result_df


import re
import pandas as pd

class LagFeatureGenerator:
    def __init__(self, lot_col='LOT_ID', target_col='discharge', n_lags=3,
                 stat_types=['mean', 'std', 'min', 'max'], fillna_value=-999):
        self.lot_col = lot_col
        self.target_col = target_col
        self.n_lags = n_lags
        self.stat_types = stat_types
        self.fillna_value = fillna_value
        self.lag_feature_names_ = []
        self.sorted_lot_groups_ = []
        self._lot_col_created = False
        self._reset_indexed = False

    def _extract_group_from_lot(self, lot):
        match = re.search(r'[ILM](\d{6})\d{2}$', lot)
        return match.group(0) if match else lot

    def _prepare_lot_col(self, df):
        df = df.copy()
        if self.lot_col not in df.columns:
            df[self.lot_col] = df.index
            self._lot_col_created = True
        if df.index.name == self.lot_col:
            df = df.reset_index()
            self._reset_indexed = True
        return df

    def _finalize_output(self, df):
        if self._lot_col_created:
            df.drop(columns=[self.lot_col], inplace=True)
        if self._reset_indexed:
            df.set_index(self.lot_col, inplace=True)
        self._lot_col_created = False
        self._reset_indexed = False
        return df

    def fit(self, df):
        df = self._prepare_lot_col(df)
        df['LOT_group'] = df[self.lot_col].apply(self._extract_group_from_lot)
        group_values = df.groupby('LOT_group')[self.target_col].mean().sort_index()
        self.sorted_lot_groups_ = group_values.index.tolist()
        return self

    def transform(self, df):
        df = self._prepare_lot_col(df)
        df['LOT_group'] = df[self.lot_col].apply(self._extract_group_from_lot)

        group_values = df.groupby('LOT_group')[self.target_col].mean()
        group_df = pd.DataFrame({self.target_col: group_values})
        group_df = group_df.reindex(self.sorted_lot_groups_)

        for stat in self.stat_types:
            for i in range(1, self.n_lags + 1):
                col_name = f'{self.target_col}_lag_{stat}_{i}'
                self.lag_feature_names_.append(col_name)

                if stat == 'mean':
                    lag_vals = [self.fillna_value]*i + group_df[self.target_col].tolist()[:-i]
                else:
                    window = group_df[self.target_col].shift(i).rolling(window=i)
                    if stat == 'std':
                        lag_vals = window.std().tolist()
                    elif stat == 'min':
                        lag_vals = window.min().tolist()
                    elif stat == 'max':
                        lag_vals = window.max().tolist()
                    else:
                        raise ValueError(f"Unsupported stat type: {stat}")
                    lag_vals[:i] = [self.fillna_value]*i

                group_df[col_name] = lag_vals

        df = df.merge(group_df[self.lag_feature_names_], left_on='LOT_group', right_index=True, how='left')
        df.drop(columns=['LOT_group'], inplace=True)
        df = self._finalize_output(df)
        return df
