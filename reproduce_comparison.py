import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATS = ['city', 'district', 'floor']

BASE_NUM = [
    'area_m2', 'rooms', 'lat', 'lng', 'extra_balcony',
    'extra_lift', 'extra_garage', 'extra_separate_kitchen',
    'extra_non_smokers_only', 'extra_air_conditioning',
    'extra_basement', 'extra_terrace', 'extra_usable_room',
    'extra_garden', 'extra_two_storey', 'has_bathtub',
    'has_garden', 'is_private'
]

GIS = [
    'distance_city_center_m', 'nearest_transit_m',
    'nearest_supermarket_m', 'nearest_university_m',
    'nearest_park_m', 'transit_count_500m', 'shops_count_1km'
]

FORBIDDEN = {
    'price', 'price_per_m2', 'details_price',
    'calculated_price_m2', 'id', 'url', 'description'
}

RAW = dict(
    iterations=500, depth=6, learning_rate=0.05,
    l2_leaf_reg=3.0, random_strength=1.0,
    loss_function='RMSE'
)

UNIT = dict(
    iterations=600, depth=6, learning_rate=0.05,
    l2_leaf_reg=8.0, random_strength=1.0,
    loss_function='RMSE'
)

MODELS = [
    'city_median_m2', 'linear_regression',
    'catboost_basic', 'catboost_text', 'catboost_gis',
    'catboost_full', 'catboost_unit', 'ensemble_50_50'
]


def metric(y, pred):
    return {
        'MAE_PLN': float(mean_absolute_error(y, pred)),
        'RMSE_PLN': float(np.sqrt(mean_squared_error(y, pred))),
        'R2': float(r2_score(y, pred))
    }


def read(path):
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path, low_memory=False)

    with path.open(encoding='utf-8-sig') as f:
        first = next((c for c in f.read(4000) if not c.isspace()), '')

    return pd.read_json(path, lines=first == '{')


def make_x(df, cols):
    x = df[cols].copy()

    for c in cols:
        if c in CATS:
            x[c] = (
                x[c].fillna('UNKNOWN')
                .astype(str)
                .replace({'nan': 'UNKNOWN', '': 'UNKNOWN'})
            )
        else:
            x[c] = pd.to_numeric(
                x[c], errors='coerce'
            ).replace([np.inf, -np.inf], np.nan).astype(float)

    return x


def linear(xtr, ytr, xte):
    nums = [c for c in xtr if c not in CATS]
    cats = [c for c in CATS if c in xtr]

    prep = ColumnTransformer([
        ('num', Pipeline([
            ('imp', SimpleImputer(
                strategy='median', keep_empty_features=True
            )),
            ('scale', StandardScaler())
        ]), nums),
        ('cat', OneHotEncoder(
            handle_unknown='ignore', sparse_output=False
        ), cats)
    ])

    mdl = Pipeline([
        ('prep', prep),
        ('lin', LinearRegression())
    ])

    mdl.fit(xtr, ytr)
    return np.maximum(0, mdl.predict(xte))


def median_city(df, tr, te):
    train = df.iloc[tr]
    test = df.iloc[te]

    values = train.price / train.area_m2
    medians = values.groupby(train.city).median()

    return (
        test.area_m2 * test.city.map(medians).fillna(values.median())
    ).to_numpy(float)


def run(args):
    df = read(args.input)

    if not df.id.is_unique or not df.url.is_unique:
        raise ValueError('Duplicate id/url')

    t_cols = sorted(c for c in df.columns if c.startswith('txt_'))
    all_cols = CATS + BASE_NUM + t_cols + GIS

    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        raise ValueError('Missing columns: ' + str(missing))

    assert (
        len(df) == 9752
        and len(t_cols) == 21
        and not FORBIDDEN.intersection(all_cols)
    ), 'Unexpected data'

    df['price'] = pd.to_numeric(df['price'], errors='raise')
    df['area_m2'] = pd.to_numeric(df['area_m2'], errors='raise')

    y = df.price.to_numpy(float)
    area = df.area_m2.to_numpy(float)

    groups = (
        df.city.astype('string').fillna('UNKNOWN')
        + ' | '
        + df.district.astype('string').fillna('UNKNOWN')
    ).to_numpy()

    seed = 42

    xviews = {
        'basic': make_x(df, CATS + BASE_NUM),
        'text': make_x(df, CATS + BASE_NUM + t_cols),
        'gis': make_x(df, CATS + BASE_NUM + GIS),
        'full': make_x(df, all_cols)
    }

    if args.validation == 'spatial':
        folds = GroupKFold(args.folds).split(df, y, groups)
    else:
        folds = KFold(
            args.folds, shuffle=True, random_state=seed
        ).split(df, y)

    args.out.mkdir(parents=True, exist_ok=True)

    fold_rows = []
    pred_rows = []
    start = time.perf_counter()

    configs = [
        ('catboost_basic', 'basic', RAW, False),
        ('catboost_text', 'text', RAW, False),
        ('catboost_gis', 'gis', RAW, False),
        ('catboost_full', 'full', RAW, False),
        ('catboost_unit', 'full', UNIT, True)
    ]

    for k, (tr, te) in enumerate(folds, 1):
        if args.validation == 'spatial':
            assert not set(groups[tr]).intersection(groups[te])

        yy = y[te]
        predictions = {
            'city_median_m2': median_city(df, tr, te),
            'linear_regression': linear(
                xviews['basic'].iloc[tr],
                y[tr],
                xviews['basic'].iloc[te]
            )
        }

        for name, key, config, unit in configs:
            model = CatBoostRegressor(
                **config,
                random_seed=seed,
                thread_count=args.threads,
                verbose=False,
                allow_writing_files=False
            )

            target = y[tr] / area[tr] if unit else y[tr]

            model.fit(
                xviews[key].iloc[tr],
                target,
                cat_features=CATS
            )

            pred = np.maximum(
                0, model.predict(xviews[key].iloc[te])
            )

            predictions[name] = pred * area[te] if unit else pred

        predictions['ensemble_50_50'] = (
            predictions['catboost_full']
            + predictions['catboost_unit']
        ) * 0.5

        for name in MODELS:
            results = metric(yy, predictions[name])

            fold_rows.append({
                'validation': args.validation,
                'fold': k,
                'model': name,
                'n': len(te),
                **results
            })

            pred_rows.append(pd.DataFrame({
                'validation': args.validation,
                'fold': k,
                'model': name,
                'id': df.id.iloc[te].to_numpy(),
                'city': df.city.iloc[te].to_numpy(),
                'price': yy,
                'prediction': predictions[name]
            }))

        print(
            f'{args.validation.upper()} fold {k}/{args.folds} '
            f'({len(te)}): '
            + ', '.join(
                f'{m}={metric(yy, predictions[m])["MAE_PLN"]:.2f}'
                for m in MODELS
            ),
            flush=True
        )

    folds_df = pd.DataFrame(fold_rows)
    predictions_df = pd.concat(pred_rows, ignore_index=True)

    summary = pd.DataFrame([
        {
            'validation': args.validation,
            'model': name,
            'n': len(data),
            **metric(data.price, data.prediction)
        }
        for name, data in predictions_df.groupby('model', sort=False)
    ]).sort_values('MAE_PLN')

    out = args.out
    folds_df.to_csv(out / f'{args.validation}_folds.csv', index=False)
    summary.to_csv(out / f'{args.validation}_summary.csv', index=False)
    predictions_df.to_csv(
        out / f'{args.validation}_oof_predictions.csv', index=False
    )

    metadata = {
        'input': str(args.input),
        'sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
        'n': len(df),
        'city_count': int(df.city.nunique()),
        'group_count': int(len(set(groups))),
        'features_basic': len(xviews['basic'].columns),
        'features_text': len(xviews['text'].columns),
        'features_gis': len(xviews['gis'].columns),
        'features_full': len(xviews['full'].columns),
        'validation': args.validation,
        'folds': args.folds,
        'seed': seed,
        'raw': RAW,
        'unit': UNIT,
        'run_seconds': round(time.perf_counter() - start, 2),
        'models': MODELS,
        'note': 'exploratory CV, no temporal holdout; no target-derived input features'
    }

    (out / f'{args.validation}_metadata.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding='utf8'
    )

    print('\nFINAL', summary.round(2).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path('rentals_v3.csv'))
    parser.add_argument('--out', type=Path, default=Path('results'))
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument(
        '--validation',
        choices=['random', 'spatial'],
        default='spatial'
    )
    parser.add_argument('--threads', type=int, default=4)

    run(parser.parse_args())
