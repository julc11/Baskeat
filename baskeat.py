from __future__ import annotations
import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# Helpers
# --------
def _parse_items_csv(s: str) -> List[str]:
    """Parse a comma-separated string into a clean list of lowercased items (keeps original case when needed)."""
    return [x.strip().strip(" '\"") for x in (s or "").split(",") if x.strip()]

def _parse_pinned_products(s: str) -> Dict[str, str]:
    """Parse 'product:store' pairs separated by commas into a dict {product_lower: store_raw}."""
    if not s:
        return {}
    out: Dict[str, str] = {}
    for chunk in s.split(","):
        part = chunk.strip().strip(" '\"")
        if not part or ":" not in part:
            continue
        prod, store = part.split(":", 1)
        k = prod.strip().strip(" '\"").lower()
        v = store.strip().strip(" '\"")
        if k and v:
            out[k] = v
    return out
# --------

# Data models
# -----------
@dataclass
class Product:
    name: str
    price: float
    quality: int  # 1..5

@dataclass
class Supermarket:
    name: str
    position: Tuple[float, float]
    home_delivery: bool = True
    delivery_cost: float = 4.99
    products: Dict[str, Product] = field(default_factory=dict)

    def get_product(self, product_name: str) -> Optional[Product]:
        return self.products.get(product_name.strip().lower())

    def has_product(self, product_name: str) -> bool:
        return product_name.strip().lower() in self.products

    def __hash__(self):
        return hash(self.name.lower())

    def __eq__(self, other):
        return isinstance(other, Supermarket) and self.name.lower() == other.name.lower()

@dataclass
class User:
    location: Tuple[float, float]
    transport: Optional[str]
    time_available: Optional[int]
    preference: str                    # 'cost' | 'quality' | 'balanced'
    favorites: List[str] = field(default_factory=list)
    mixed_shopping: bool = True
    pinned_products: Dict[str, str] = field(default_factory=dict)  # {product_lower: store_name}
    allowed_transports: List[str] = field(default_factory=list)     # e.g., ['car','bus','walk','delivery']
    per_store_mixed_transport: bool = False
    lambda_eur_per_min: float = 0.0    # time penalty €/min
    max_stores: Optional[int] = None   # limit number of stores

    def __post_init__(self):
        self.favorites = [f.strip().lower() for f in (self.favorites or [])]
        self.pinned_products = {k.strip().lower(): v.strip() for k, v in (self.pinned_products or {}).items()}
        self.allowed_transports = [t.strip().lower() for t in (self.allowed_transports or []) if t.strip()]
        if self.transport and not self.allowed_transports:
            self.allowed_transports = [self.transport]

# -----------

# System parameters
# ---------
SPEED_KMH = {'car': 50.0, 'bus': 30.0, 'walk': 5.0, 'delivery': 0.0}
COST_PER_KM_CAR = 0.20
BUS_FARE = 1.95

# More realistic shopping time model
TIME_PER_ITEM_MIN = 0.5     # minutes per item (~30s)
OVERHEAD_PER_STORE_MIN = 5.0 # fixed minutes per stop (enter, pay, etc.)

# Delivery minimum rules per store
DELIVERY_MIN_TOTAL = 15.0
DELIVERY_MIN_ITEMS = 3

# ---------

# Distances / times
# ------------
def distance(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0]-b[0], a[1]-b[1])

def sort_by_proximity(origin: Tuple[float,float], stores: List['Supermarket']) -> List['Supermarket']:
    """Sort stores by distance to origin."""
    return sorted(stores, key=lambda s: distance(origin, s.position))

def _round_trip_route(origin: Tuple[float,float], stores: List['Supermarket']) -> float:
    """Heuristic route length: nearest-neighbor order + return to origin."""
    if not stores:
        return 0.0
    order = sort_by_proximity(origin, stores)
    total = distance(origin, order[0].position)
    for i in range(len(order)-1):
        total += distance(order[i].position, order[i+1].position)
    total += distance(order[-1].position, origin)
    return total

def _time_and_cost_for_group(origin: Tuple[float,float],
                             stores: List['Supermarket'],
                             transport: str,
                             items_per_store: Optional[Dict[str,int]] = None,
                             subtotals_per_store: Optional[Dict[str,float]] = None
                             ) -> Tuple[float,float,Dict[str,float]]:
    """
    Time (min) and transport cost (€) for visiting 'stores' with 'transport' from 'origin'.
    Returns (total_time_min, transport_or_delivery_cost_eur, delivery_fee_per_store).
    For delivery: time = 0, cost = sum of delivery fees per store (if delivery minimums are met).
    """
    delivery_fee_per_store: Dict[str, float] = {}
    if not stores:
        return 0.0, 0.0, delivery_fee_per_store

    if transport == 'delivery':
        # All stores must allow delivery and meet per-store minimums
        for s in stores:
            if not s.home_delivery:
                return float('inf'), float('inf'), {}
            n_items = (items_per_store or {}).get(s.name, 0)
            amount = (subtotals_per_store or {}).get(s.name, 0.0)
            if (n_items < DELIVERY_MIN_ITEMS) or (amount < DELIVERY_MIN_TOTAL):
                return float('inf'), float('inf'), {}
            delivery_fee_per_store[s.name] = s.delivery_cost
        cost = round(sum(delivery_fee_per_store.values()), 2)
        return 0.0, cost, delivery_fee_per_store

    # Distance and travel time
    dist = _round_trip_route(origin, stores)
    v = SPEED_KMH.get(transport, 0.0)
    time_min = (dist / v) * 60.0 if v > 0 else 0.0

    # Overheads: per store and per item
    time_min += OVERHEAD_PER_STORE_MIN * len(stores)
    total_items = sum((items_per_store or {}).get(s.name, 0) for s in stores)
    time_min += TIME_PER_ITEM_MIN * total_items

    # Transport cost
    if transport == 'car':
        cost = COST_PER_KM_CAR * dist
    elif transport == 'bus':
        cost = BUS_FARE * (len(stores) + 1)  # simple model: legs incl. return
    elif transport == 'walk':
        cost = 0.0
    else:
        cost = 0.0

    return round(time_min, 1), round(cost, 2), delivery_fee_per_store

# ----------


# Build product plans
# -----------
def _subsets_non_empty(stores: List['Supermarket']) -> List[List['Supermarket']]:
    """All non-empty subsets of stores."""
    return [list(c) for r in range(1, len(stores)+1) for c in itertools.combinations(stores, r)]

def _conditional_assignment(subset: List['Supermarket'],
                            user: User,
                            shopping_list: List[str]) -> Optional[Dict['Supermarket', List[str]]]:
    """
    Assign products **within** the given subset respecting pinned items and user preference.
    Returns a plan {store: [items]} or None if not feasible.
    """
    plan: Dict[Supermarket, List[str]] = {}

    # First: pinned products (must-buy at a specific store)
    for item in shopping_list:
        pinned = user.pinned_products.get(item.lower())
        if pinned:
            pinned_store = next((s for s in subset if s.name.lower() == pinned.lower()), None)
            if (pinned_store is None) or (not pinned_store.has_product(item)):
                return None
            plan.setdefault(pinned_store, []).append(item)

    # Then: rest of products by user preference
    for item in shopping_list:
        if item.lower() in user.pinned_products:
            continue
        chosen: Optional[Supermarket] = None
        pref = user.preference

        # If item is a favorite OR 'quality' mode -> pick highest quality
        if item.lower() in user.favorites or pref == 'quality':
            best_q = -1
            for s in subset:
                p = s.get_product(item)
                if p and p.quality > best_q:
                    best_q = p.quality
                    chosen = s

        # If still not chosen and pref is 'cost' -> pick lowest price
        if chosen is None and pref == 'cost':
            best_price = float('inf')
            for s in subset:
                p = s.get_product(item)
                if p and p.price < best_price:
                    best_price = p.price
                    chosen = s

        # If still not chosen and pref is 'balanced' -> best (quality/price)
        if chosen is None and pref == 'balanced':
            best_ratio = -1.0
            for s in subset:
                p = s.get_product(item)
                if p:
                    ratio = (p.quality/5.0) / (p.price + 1e-3)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        chosen = s

        if chosen is None:
            return None

        plan.setdefault(chosen, []).append(item)

    return plan

def _cost_and_quality(plan: Dict['Supermarket', List[str]]) -> Tuple[float, Dict[str,float], float, int]:
    """Return (products_total_cost, per_store_subtotals, total_quality_points, n_items)."""
    subtotals: Dict[str, float] = {}
    total_cost = 0.0
    total_quality = 0
    n_items = 0
    for s, items in plan.items():
        st = 0.0
        for it in items:
            p = s.get_product(it)
            if not p:
                return float('inf'), {}, 0.0, 0
            st += p.price
            total_quality += p.quality
            n_items += 1
        subtotals[s.name] = round(st, 2)
        total_cost += st
    return round(total_cost, 2), subtotals, float(total_quality), n_items

# --------------

# Evaluations with flags & per-store delivery
# ------------------
def _evaluate_global_transport(plan: Dict['Supermarket', List[str]],
                               user: User,
                               transport: str) -> Optional[Dict]:
    stores = [s for s, items in plan.items() if items]
    products_cost, subt, sum_q, n_items = _cost_and_quality(plan)
    if not n_items or products_cost == float('inf'):
        return None

    items_count = {s.name: len(plan.get(s, [])) for s in stores}
    time_min, trans_cost, delivery_by_store = _time_and_cost_for_group(
        user.location, stores, transport, items_count, subt
    )
    if math.isinf(time_min) or math.isinf(trans_cost):
        return None

    violates_time = (
        user.time_available is not None
        and transport != 'delivery'
        and time_min > user.time_available
    )

    avg_quality = round(sum_q / n_items, 2) if n_items else 0.0
    total = round(products_cost + trans_cost, 2)
    composed = round(total + user.lambda_eur_per_min * time_min, 2)

    return {
        "per_store_subtotals": subt,
        "items_per_store": {s.name: items[:] for s, items in plan.items() if items},
        "delivery_fee_per_store": delivery_by_store,
        "products_total_cost": products_cost,
        "estimated_time_min": time_min,
        "transport_or_delivery_cost": trans_cost,
        "avg_quality": avg_quality,
        "stores": [s.name for s in stores],
        "estimated_total": total,
        "composed_cost": composed,
        "transport": transport,
        "violates_time": violates_time,
    }

def _evaluate_mixed_transport(plan: Dict['Supermarket', List[str]],
                              user: User,
                              allowed_transports: List[str]) -> List[Dict]:
    stores = [s for s, items in plan.items() if items]
    if not stores:
        return []
    products_cost, subt, sum_q, n_items = _cost_and_quality(plan)
    if not n_items or products_cost == float('inf'):
        return []
    results: List[Dict] = []
    for combo in itertools.product(allowed_transports, repeat=len(stores)):
        mapping = {stores[i]: combo[i] for i in range(len(stores))}
        groups: Dict[str, List[Supermarket]] = {}
        for s, tr in mapping.items():
            groups.setdefault(tr, []).append(s)

        total_time = 0.0
        total_trans_cost = 0.0
        delivery_by_store: Dict[str, float] = {}
        viable = True
        for tr, group_stores in groups.items():
            items_count = {s.name: len(plan.get(s, [])) for s in group_stores}
            subts = {s.name: subt[s.name] for s in group_stores if s.name in subt}
            t_min, c_eur, env_dict = _time_and_cost_for_group(
                user.location, group_stores, tr, items_count, subts
            )
            if math.isinf(t_min) or math.isinf(c_eur):
                viable = False
                break
            total_time += t_min
            total_trans_cost += c_eur
            delivery_by_store.update(env_dict)
        if not viable:
            continue

        violates_time = (
            user.time_available is not None
            and total_time > user.time_available
        )

        avg_quality = round(sum_q / n_items, 2)
        total = round(products_cost + total_trans_cost, 2)
        composed = round(total + user.lambda_eur_per_min * total_time, 2)

        results.append({
            "per_store_subtotals": subt,
            "items_per_store": {s.name: plan[s][:] for s in plan if plan[s]},
            "delivery_fee_per_store": delivery_by_store,
            "products_total_cost": products_cost,
            "estimated_time_min": round(total_time, 1),
            "transport_or_delivery_cost": round(total_trans_cost, 2),
            "avg_quality": avg_quality,
            "stores": [s.name for s in stores],
            "estimated_total": total,
            "composed_cost": composed,
            "transport_by_store": {s.name: mapping[s] for s in stores},
            "violates_time": violates_time,
        })
    return results

# -------------

# Ranking
# ------------------
def _order_candidates(cands: List[Dict],
                      preference: str,
                      delta_eur: float = 1.0,
                      use_composed: bool = False) -> List[Dict]:
    if not cands:
        return []
    key_cost = ("composed_cost" if use_composed else "estimated_total")
    if preference == 'cost':
        cands = sorted(cands, key=lambda ev: ev[key_cost])
        min_cost = cands[0][key_cost]
        group = [ev for ev in cands if ev[key_cost] <= min_cost + delta_eur]
        group = sorted(group, key=lambda ev: (ev["estimated_time_min"], len(ev.get("stores", []))))
        rest = [ev for ev in cands if ev[key_cost] > min_cost + delta_eur]
        return group + rest
    elif preference == 'quality':
        return sorted(cands, key=lambda ev: (-ev["avg_quality"], ev[key_cost], ev["estimated_time_min"]))
    else:  # balanced
        by_cost = sorted(cands, key=lambda ev: ev[key_cost])
        by_qual = sorted(cands, key=lambda ev: ev["avg_quality"], reverse=True)
        r_cost = {id(ev): i+1 for i, ev in enumerate(by_cost)}
        r_qual = {id(ev): i+1 for i, ev in enumerate(by_qual)}
        return sorted(cands, key=lambda ev: (r_cost[id(ev)] + r_qual[id(ev)], ev["estimated_time_min"]))

# ----------

# Explain unfeasibility
# ----------------
def _unfeasibility_summary(candidates: List[Dict], user: User) -> List[str]:
    msgs = []
    if not candidates:
        msgs.append("Could not generate any plans (possibly due to pinned products incompatibility or missing items in catalogs).")
        return msgs
    min_time = min(ev["estimated_time_min"] for ev in candidates)
    if user.time_available is not None and all(ev.get("violates_time", False) for ev in candidates):
        msgs.append(
            f"No plan satisfies your maximum time ({user.time_available} min). "
            f"The minimum achievable time with your inputs is {min_time:.1f} min."
        )
    min_cost = min(ev["estimated_total"] for ev in candidates)
    msgs.append(f"Best alternative ignoring the time constraint: minimum possible cost €{min_cost:.2f}.")
    return msgs

# --------------

# Core engine
# -----------------
def recommend(user: User,
              supermarkets: List[Supermarket],
              shopping_list: List[str],
              delta_eur: float = 1.0,
              top_k: int = 5) -> Dict[str, List[Dict]]:
    results: Dict[str, List[Dict]] = {}
    subsets = _subsets_non_empty(supermarkets)

    # Build feasible plans per subset
    plans = []
    for subset in subsets:
        plan = _conditional_assignment(subset, user, shopping_list)
        if plan is None:
            continue
        # Limit number of stores
        if user.max_stores is not None and len([s for s, items in plan.items() if items]) > user.max_stores:
            continue
        plans.append(plan)
    if not plans:
        return results

    use_composed = user.lambda_eur_per_min > 0.0

    # Mode without mixed shopping: exactly one store
    if not user.mixed_shopping:
        for tr in (user.allowed_transports or ['car']):
            all_candidates = []
            for plan in plans:
                if len(plan.keys()) != 1:
                    continue
                ev = _evaluate_global_transport(plan, user, tr)
                if ev:
                    all_candidates.append(ev)

            if user.time_available is not None and tr != 'delivery':
                ok_candidates = [ev for ev in all_candidates if not ev.get("violates_time", False)]
            else:
                ok_candidates = all_candidates[:]

            if ok_candidates:
                ranking = _order_candidates(ok_candidates, user.preference, delta_eur, use_composed)
                if ranking:
                    results[tr] = ranking[:min(top_k, len(ranking))]
            elif all_candidates:
                ranking = _order_candidates(all_candidates, user.preference, delta_eur, use_composed)
                if ranking:
                    results[tr] = ranking[:min(top_k, len(ranking))]
                    results.setdefault('_warnings', []).extend(_unfeasibility_summary(all_candidates, user))
        return results

    # Mixed shopping
    if user.per_store_mixed_transport and (len(user.allowed_transports) > 1):
        all_candidates = []
        for plan in plans:
            all_candidates.extend(_evaluate_mixed_transport(plan, user, user.allowed_transports))

        if user.time_available is not None:
            ok_candidates = [ev for ev in all_candidates if not ev.get("violates_time", False)]
        else:
            ok_candidates = all_candidates[:]

        if ok_candidates:
            ranking = _order_candidates(ok_candidates, user.preference, delta_eur, use_composed)
            if ranking:
                results['mixed'] = ranking[:min(top_k, len(ranking))]
        elif all_candidates:
            ranking = _order_candidates(all_candidates, user.preference, delta_eur, use_composed)
            if ranking:
                results['mixed'] = ranking[:min(top_k, len(ranking))]
                results.setdefault('_warnings', []).extend(_unfeasibility_summary(all_candidates, user))
        return results
    else:
        for tr in (user.allowed_transports or ['car']):
            all_candidates = []
            for plan in plans:
                ev = _evaluate_global_transport(plan, user, tr)
                if ev:
                    all_candidates.append(ev)

            if user.time_available is not None and tr != 'delivery':
                ok_candidates = [ev for ev in all_candidates if not ev.get("violates_time", False)]
            else:
                ok_candidates = all_candidates[:]

            if ok_candidates:
                ranking = _order_candidates(ok_candidates, user.preference, delta_eur, use_composed)
                if ranking:
                    results[tr] = ranking[:min(top_k, len(ranking))]
            elif all_candidates:
                ranking = _order_candidates(all_candidates, user.preference, delta_eur, use_composed)
                if ranking:
                    results[tr] = ranking[:min(top_k, len(ranking))]
                    results.setdefault('_warnings', []).extend(_unfeasibility_summary(all_candidates, user))
        return results
#------

# Simulated data
#------

PRODUCT_NAMES = [
    "Milk","Bread","Eggs","Apples","Bananas","Chicken","Beef","Pasta","Rice",
    "Tomatoes","Potatoes","Onions","Lettuce","Carrots","Yogurt","Cheese","Cereal","Tuna",
    "Olive oil","Coffee","Tea","Sugar","Salt","Flour","Butter","Jam",
    "Cookies","Soy yogurt","Cow cheese","Hake"
]

def generate_catalog(seed: int, price_min: float, price_max: float) -> Dict[str, Product]:
    random.seed(seed)
    catalog: Dict[str, Product] = {}
    for name in PRODUCT_NAMES:
        price = round(random.uniform(price_min, price_max), 2)
        quality = random.randint(1, 5)
        catalog[name.lower()] = Product(name=name, price=price, quality=quality)
    return catalog

def create_supermarkets() -> List[Supermarket]:
    dia = Supermarket("DIA", (1.0, 3.0), True, 4.99, generate_catalog(10, 0.7, 9.5))
    eroski = Supermarket("Eroski", (3.0, 0.0), True, 3.99, generate_catalog(20, 0.6, 9.0))
    alcampo = Supermarket("Alcampo", (0.0, 1.0), True, 5.49, generate_catalog(30, 0.8, 10.5))
    return [dia, eroski, alcampo]
#----

# Pretty printing

#-------
def _print_option(ev: Dict, idx: int, lambda_penalty: float):
    head = (
        f"{idx}) Stores: {', '.join(ev['stores'])} — "
        f"Total: €{ev['estimated_total']:.2f} — "
        f"Time: {ev['estimated_time_min']} min — "
        f"Quality: {ev['avg_quality']:.2f}"
    )
    if lambda_penalty > 0:
        head += f" — Comp(λ={lambda_penalty:.2f}): €{ev['composed_cost']:.2f}"
    if 'transport' in ev:
        head += f" — Transport: {ev['transport']}"
    if 'transport_by_store' in ev:
        mapping = ", ".join([f"{k}:{v}" for k, v in ev['transport_by_store'].items()])
        head += f" — Transports: {mapping}"
    if ev.get("violates_time"):
        head += "  ( exceeds your max time)"
    print(head)

    deliveries = ev.get("delivery_fee_per_store", {})
    print("   Per-store breakdown:")
    for store, cost in ev["per_store_subtotals"].items():
        items = ", ".join(ev.get("items_per_store", {}).get(store, []))
        delivery_txt = f" + delivery €{deliveries[store]:.2f}" if store in deliveries else ""
        print(f"     · {store}: products €{cost:.2f}{delivery_txt} — {items}")

def print_results(results: Dict[str, List[Dict]], lambda_penalty: float):
    if not results:
        print("No feasible plans for the given constraints.")
        return

    # Warnings
    warnings = results.get('_warnings', [])
    if warnings:
        print("\n Warnings:")
        for w in warnings:
            print(" -", w) 

    for block, ranking in results.items():
        if block == '_warnings':
            continue
        title = "Mixed (store→transport)" if block == 'mixed' else f"Global transport: {block.upper()}"
        print("\n===", title, "===")
        for i, ev in enumerate(ranking, 1):
            _print_option(ev, i, lambda_penalty)
#--------

# CLI
#-----------

def cli():
    supermarkets = create_supermarkets()
    print("Welcome to the intelligent grocery planner")
    print("Supermarkets:", ", ".join(s.name for s in supermarkets))

    loc = input("Your coordinates x,y on the map (0-3,0-3) [0,0]: ").strip() or "0,0"
    try:
        x_str, y_str = loc.split(",")
        location = (float(x_str), float(y_str))
    except Exception:
        print("Invalid format. Using (0,0)."); location = (0.0, 0.0)

    preference = (input("Preference (cost/quality/balanced) [cost]: ").strip().lower() or "cost")
    tdisp = input("Time available in minutes (blank = no limit): ").strip()
    time_available = int(tdisp) if tdisp else None
    mixed_shopping = (input("Allow mixed shopping (multiple stores)? (y/n) [y]: ").strip().lower() or "y") == "y"

    transp = input("Allowed transports (car,bus,walk,delivery) [car]: ").strip()
    allowed_transports = _parse_items_csv(transp or "car")

    mixed_flag = False
    if mixed_shopping and len(allowed_transports) > 1:
        mixed_flag = (input("Allow different transport per store? (y/n) [y]: ").strip().lower() or "y") == "y"

    favs = input("Favorite products (comma) [blank]: ").strip()
    favorites = _parse_items_csv(favs)
    favorites = [f.lower() for f in favorites]

    pinned = input("Pinned products (product:store, comma) [blank]: ").strip()
    pinned_products = _parse_pinned_products(pinned)

    delta_str = input("Price indifference threshold (€/diff) to favor time [1.0]: ").strip()
    try:
        delta_eur = float(delta_str) if delta_str else 1.0
    except Exception:
        delta_eur = 1.0

    lam_str = input("Penalty λ in €/min for composite cost (0 disables) [0.2]: ").strip()
    try:
        lambda_penalty = float(lam_str) if lam_str else 0.2
    except Exception:
        lambda_penalty = 0.2

    max_st_str = input("Maximum number of stores allowed (blank = no limit): ").strip()
    max_stores = int(max_st_str) if max_st_str else None

    list_str = input("Shopping list (comma): ").strip()
    shopping_list = _parse_items_csv(list_str)

    user = User(
        location=location,
        transport=None,
        time_available=time_available,
        preference=preference,
        favorites=favorites,
        mixed_shopping=mixed_shopping,
        pinned_products=pinned_products,
        allowed_transports=allowed_transports,
        per_store_mixed_transport=mixed_flag,
        lambda_eur_per_min=lambda_penalty,
        max_stores=max_stores,
    )

    results = recommend(user, supermarkets, shopping_list, delta_eur=delta_eur, top_k=7 if mixed_shopping else 3)
    print_results(results, lambda_penalty)
#--------
# Demo

#-------

def demo():
    supermarkets = create_supermarkets()
    user = User(
        location=(0.0, 0.0),
        transport=None,
        time_available=20,
        preference='cost',
        favorites=['yogurt'],
        mixed_shopping=True,
        pinned_products={'hake': 'Alcampo'},
        allowed_transports=['walk', 'car', 'delivery'],
        per_store_mixed_transport=True,
        lambda_eur_per_min=0.2,
        max_stores=2
    )
    shopping_list = ['Milk','Apples','Cow cheese','Hake','Cookies','Soy yogurt','Salt','Flour','Tea','Cereal']
    results = recommend(user, supermarkets, shopping_list, delta_eur=1.0, top_k=5)
    print_results(results, user.lambda_eur_per_min)

if __name__ == "__main__":
    cli()  # change to demo() to see the automatic example
