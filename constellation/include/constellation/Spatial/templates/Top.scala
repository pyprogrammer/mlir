R"CONST(
import spatial.dsl._

@spatial object {{ name }} extends SpatialApp {
    def main(args: Array[String]): Unit = {

{{ preAccel }}

        Accel * {
{{ accel }}
        }

{{ postAccel }}

    }
}
)CONST"
