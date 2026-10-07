# IPC Message Transport Protocol

## Structure

Each message is sent in a packet that contains

| size | data-type | name           | description                         |
| ---- | --------- | -------------- | ----------------------------------- |
| `4`  | `uint`    | version        | Protocol version (default=1)        |
| `2`  | `ushort`  | message-type   | Payload schema description          |
| `2`  | `ushort`  | payload-length | Payload length                      |
| ...  | `string`  | json-payload   | Message data encoded as a JSON blob |

## Communication

### Initialization

  1. Client sends an `INIT_REQ` payload: [InitRequest](#initrequest)
  2. Server responds with `INIT_RESP` payload: [InitResponse](#initresponse)
  3. If initialization fails, an `ERROR_RESP` is returned instead: [ErrorResponse](#errorresponse)

### Get Status

  1. Client sends a `STATUS_REQ` message (no payload).
  2. Server responds with `STATUS_RESP` payload: [StatusResponse](#statusrequest)

Note that the status is returned as the response to many other message types. See below.

### Sync Shared Memory

  1. Client sends a `SYNC_REQ` message (no payload).
  2. Server responds with `STATUS_RESP` payload: [StatusResponse](#statusrequest)

### Pause

  1. Client sends a `CLOCK_UPDATE` payload: [ClockUpdateRequest](#clockupdaterequest)
     - Field `running` set to `false`
  2. Server responds with `STATUS_RESP` payload: [StatusResponse](#statusrequest)

### Unpause

  1. Client sends a `CLOCK_UPDATE` payload: [ClockUpdateRequest](#clockupdaterequest)
     - Field `running` set to `true`
  2. Server responds with `STATUS_RESP` payload: [StatusResponse](#statusrequest)

### Set Speed

  1. Client sends a `CLOCK_UPDATE` payload: [ClockUpdateRequest](#clockupdaterequest)
     - Field `speed` set to a `float` value greater than `0.0`
  2. Server responds with `STATUS_RESP` payload: [StatusResponse](#statusrequest)

### Shift Vectors

  1. Client sends a `SHIFT_VECTOR_REQ` payload: [ShiftVectorsRequest](#shiftvectorsrequest)
     - Field `vector_name`: one of `position`, `velocity`, `mass`, or `radius`
     - Field `ids`: Array of body IDs
     - Field `op`: Set to `add` to increment/decrement values. Set to `mul` to multiply.
     - Field `offset`: Tuple of `(x, y)` offsets to apply. Offsets for `float` vectors always have `y` = `0`.
  2. Server responds with `STATUS_RESP` payload: [StatusResponse](#statusrequest)

## Message Schemas

### ErrorResponse

    {
        success: boolean
        error_message: string
    }

### InitRequest

    {
        device: {
            device_id: uint,               // OpenCL device ID 
            platform_id: uint,             // OpenCL platform ID
        }
        particles: [
            {
                flags: uint,
                position: [float, float],  // (x, y)
                velocity: [float, float],  // (x, y)
                mass: float, 
                radius: float
            },
            ... 
        ],
        reinit: true                       // will force a re-init if already running
    }

### InitResponse

    {
        initialized: boolean
        config: {
            G: float
            EPS_DIST: float
            EPS_TIME: float
            coef_of_restitution: float
            dt_base: float
            N: uint
        },
        memory: {
            vector_name: {
                name: string
                dtype: string
                size: string
                shape: [int, ...]
            },
            ...
        }
    }

### StatusRequest

    {
        initialized: boolean
        tick_id: uint
        accum: float
        clock: {
            duration: float
            running: boolean
            last_time_ms: float
            speed: float
        }
    }

### ShiftVectorsRequest

    {
        vector_name: string
        ids: [uint, ...]
        op: "add" | "mul"
        offset: [float, float]
    }

### ClockUpdateRequest

    {
        speed: float|null
        running: boolean|null
    }
