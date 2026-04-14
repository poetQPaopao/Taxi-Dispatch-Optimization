

def flatten_loc(loc, grid_size):
    '''
    Env: 
        loc: [x, y]
    Agent:
        location: flatten zone id -> int
    
    Align with the agent's location input; Each location has one unique index.
    '''
    # already flattened
    if isinstance(loc, int):
        return loc
    
    # list / tuple / np.array
    if isinstance(loc, (list, tuple)):
        x, y = loc
        return int(x) * grid_size + int(y)
    

def adapt_obs(obs, grid_size):
    '''
    Align the format of "observation" from Env and Agent,
    adapt them into a unified form.

    return: observation
    '''
    return {
        "taxis": [
            {
                "id": t["id"],
                "location": flatten_loc(t["location"], grid_size),
                "is_free": t["is_free"],
            }
            for t in obs["taxis"]
        ],
        "orders": [
            {
                "id": o["id"],
                "pickup": flatten_loc(o["pickup"], grid_size),
                "created_time": o["created_time"],
            }
            for o in obs["orders"]
        ],
        "current_time": obs["current_time"],
    }


def get_pending_order_ids(obs):
    '''
    Contract: env.obs['orders'] contains unfinished orders only;

    return: a pending order id list 
    '''
    return [o["id"] for o in obs["orders"]]