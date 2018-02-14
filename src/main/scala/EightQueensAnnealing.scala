import scala.util.Random
import scala.math.exp

/**
  * Solve the 8-Queens or 8-Rocks problems using a Neural network (continuous Ising model)
  */
object EightQueensAnnealing {
  def main(args: Array[String]): Unit = {
    val board = new Board(false)
    for(i <- 0 to 1000) {
      // Show progress
      if( i % 10 == 0 ) {
        println(s"Iteration: $i\ttemp:${board.temp}\tok:${board.isOk}\n$board")
        if(board.isOk()) return
      }

      board.eval()
      board.evolveTemp()
    }
  }
}

/**
  * 8 queens demo
  */
class Board(val rocks: Boolean) {
  val bias = 1.0
  val BOARD_SIZE = 8  // Board size
  val INIT_WEIGHT = 1.0 // Initial outputs range
  var temp = 1.0
  val tempConstant = 0.99
  val ACTIVE_THRESHOLD = 0.1
  val out= Array.ofDim[Double](BOARD_SIZE, BOARD_SIZE)
  val rand= new Random(20171207)

  /** Initialize using random outputs */
  for(i <- 0 until BOARD_SIZE; j <- 0 until BOARD_SIZE)
    out(i)(j) = rand.nextDouble() * INIT_WEIGHT

  /** Count 'active' entries in the board */
  def countActive(): Int = {
    var count = 0
    for(i <- 0 until BOARD_SIZE; j <- 0 until BOARD_SIZE; if isActive(i, j)) { count += 1 }
    count
  }

  def eval(): Unit = {
    for (_ <- 0 until BOARD_SIZE*BOARD_SIZE) {
      val xi = rand.nextInt(BOARD_SIZE)
      val yi = rand.nextInt(BOARD_SIZE)
      out(xi)(yi) = y(xi, yi)
    }
  }

  def evolveTemp() : Double = {
    temp = temp * tempConstant
    temp
  }

  /** Activation */
  def h(x: Int, y: Int): Double = {
    bias - (if (rocks) hrocks(x, y) else hrocks(x, y) + hdiag(x, y))
  }

  def hdiag(x: Int, y: Int) : Double = {
    var sum = 0.0

    for(i <- 1 until BOARD_SIZE; xi=x-i; yi=y-i; if xi >= 0 && yi >= 0)
      sum += out(xi)(yi)

    for(i <- 1 until BOARD_SIZE; xi=x+i; yi=y+i; if xi < BOARD_SIZE && yi < BOARD_SIZE)
      sum += out(xi)(yi)

    for(i <- 1 until BOARD_SIZE; xi=x-i; yi=y+i; if xi >= 0 && yi < BOARD_SIZE)
      sum += out(xi)(yi)

    for(i <- 1 until BOARD_SIZE; xi=x+i; yi=y-i; if xi < BOARD_SIZE && yi >= 0)
      sum += out(xi)(yi)

    sum
  }

  def hrocks(x: Int, y: Int): Double = {
    var sum = 0.0

    for (i <- 0 until BOARD_SIZE; if i != x)
      sum += out(i)(y)

    for (j <- 0 until BOARD_SIZE; if j != y)
      sum += out(x)(j)

    sum
  }

  /** Is this coordinate within the board? */
  def inBoard(i: Int, j: Int): Boolean = (i >= 0) && (i < BOARD_SIZE) && (j >= 0) && (j < BOARD_SIZE)

  def isActive(i: Int, j: Int) : Boolean = out(i)(j) >= ACTIVE_THRESHOLD

  /** Is the board OK? Does the solution check */
  def isOk() : Boolean = (countActive() == BOARD_SIZE) && isOkMain() && (rocks || isOkDiagonals())

  /** Is the board OK? Check 'main' lines */
  def isOkMain() : Boolean = {
    for (i <- 0 until BOARD_SIZE) {
      val countRow = (0 until BOARD_SIZE).map(j => isActive(i, j)).count(a => a)
      val countCol = (0 until BOARD_SIZE).map(j => isActive(j, i)).count(a => a)
      if (countRow > 1 && countCol > 1) return false
    }
    true
  }

  /** Is the board OK? Check 'diagonal' lines */
  def isOkDiagonals() : Boolean = {
    for (i <- 0 until BOARD_SIZE) {
      var count = 0
      for(j <- 0 until BOARD_SIZE; xj = i+j; yj = j; if inBoard(xj, yj) && isActive(xj, yj)) { count += 1 }
      if(count > 1) return false

      count = 0
      for(j <- 0 until BOARD_SIZE; xj = i-j; yj = j; if inBoard(xj, yj) && isActive(xj, yj)) { count += 1 }
      if(count > 1) return false
    }
    true
  }

  override def toString: String = {
    var str = ""
    for (i <- 0 until BOARD_SIZE) {
      str += "|"
      for (j <- 0 until BOARD_SIZE) {
        str += (if(isActive(i, j)) "*" else " ")
      }
      str += "|\n"
    }
    str
  }

  def toStringFull: String = {
    var str = ""
    for (i <- 0 until BOARD_SIZE) {
      for (j <- 0 until BOARD_SIZE) {
        str += f"${out(i)(j)}%.2f "
      }
      str += "\n"
    }
    str
  }
  def y(x: Int, y: Int): Double = 1.0 / (1.0 + exp(- h(x,y) / temp))
}