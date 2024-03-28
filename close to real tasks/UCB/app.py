import uvicorn
from fastapi import FastAPI, Query
import numpy as np

offer_shows = {}
offer_actions = {}
clicks_to_offers = {}
reward_for_offer = {}

STEP = 0

app = FastAPI()


@app.on_event("shutdown")
async def shutdown_event():
    offer_shows.clear()
    offer_actions.clear()
    clicks_to_offers.clear()
    reward_for_offer.clear()


@app.put("/feedback/")
def feedback(click_id: int, reward: float = Query(ge=0)) -> dict:
    """Get feedback for particular click"""
    offer_id = clicks_to_offers[click_id]
    is_conversion = reward != 0
    response = {
        "click_id": click_id,
        "offer_id": offer_id,
        "is_conversion": is_conversion,
        "reward": reward
    }

    offer_actions[offer_id] += 1 if is_conversion else 0
    reward_for_offer[offer_id] += reward

    return response


@app.get("/offer_ids/{offer_id}/stats/")
def stats(offer_id: int) -> dict:
    """Return offer's statistics"""
    if offer_id in offer_shows:
        response = {
            "offer_id": offer_id,
            "clicks": offer_shows[offer_id],
            "conversions": offer_actions[offer_id],
            "reward": reward_for_offer[offer_id],
            "cr": (offer_actions[offer_id] / offer_shows[offer_id]) if offer_shows[offer_id] != 0 else 0.0,
            "rpc": (reward_for_offer[offer_id] / offer_shows[offer_id]) if offer_shows[offer_id] != 0 else 0.0
        }
    else:
        response = {
            "offer_id": offer_id,
            "clicks": 0,
            "conversions": 0,
            "reward": 0.0,
            "cr": 0.0,
            "rpc": 0.0
        }
    return response


@app.get("/sample/")
def sample(click_id: int, offer_ids: str) -> dict:
    """Greedy sampling"""
    eps = np.random.rand()
    global STEP

    offers_ids = [int(offer) for offer in offer_ids.split(",")]

    if eps < 0.1:
        best_offer = offers_ids[0]
        best_rpc = 0.0
        for offer_id in offers_ids:
            if offer_id in offer_shows and reward_for_offer[offer_id] != 0:
                rpc = reward_for_offer[offer_id] / offer_shows[offer_id]
                if rpc > best_rpc:
                    best_offer = offer_id
                    best_rpc = rpc
        response = {
            "click_id": click_id,
            "offer_id": best_offer,
        }
        STEP += 1
    else:
        best_offer = offers_ids[0]
        best_ucb = 0.0
        for offer_id in offers_ids:
            if offer_id in offer_shows:
                rpc = reward_for_offer[offer_id] / offer_shows[offer_id]
                # rew = reward_for_offer[offer_id]
                ucb = rpc + np.sqrt(np.log(STEP) / offer_shows[offer_id])
                if ucb > best_ucb:
                    best_offer = offer_id
                    best_ucb = ucb
            else:
                best_offer = offer_id
                break
        response = {
            "click_id": click_id,
            "offer_id": best_offer,
        }
        STEP += 1

    clicks_to_offers[click_id] = best_offer

    if best_offer not in offer_shows:
        offer_shows[best_offer] = 0
    offer_shows[best_offer] += 1

    if best_offer not in offer_actions:
        offer_actions[best_offer] = 0

    if best_offer not in reward_for_offer:
        reward_for_offer[best_offer] = 0.0
    return response


def main() -> None:
    """Run application"""
    uvicorn.run("main:app", host="localhost")


if __name__ == "__main__":
    main()
