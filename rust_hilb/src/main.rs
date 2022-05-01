use rhai::{Engine, EvalAltResult};

pub fn main() -> Result<(), Box<EvalAltResult>>
{
    let engine = Engine::new();

    let result = engine.eval::<i64>("40 + 2")?;
    //                      ^^^^^^^ required: cast the result to a type

    println!("Answer: {}", result);             // prints 42

    let num = 4 as f64;
    println!("Relu: {}", relu(num));

    let mut xs = vec![1.0, 3.0, 4.0, 5.0];
    xs = softmax(xs);
    println!("0: {}, 1: {}, 2: {}, 3: {}", xs[0], xs[1], xs[2], xs[3]);

    println!("vec1={:?}", xs);
    
    let mut inp = vec![1.0,0.0];
    let w = vec![vec![1.0,2.0], vec![3.0,1.0], vec![0.0,1.0]];
    let b = vec![9.0,8.0,-10.0];
    inp = fc(inp, &w, &b);
    inp = fc(inp, &w, &b);
    println!("res: {:?}", inp);

    let mut x = vec![vec![vec![0.0, 9.1e-3], vec![1.0, 0.0], vec![0.0, 0.0], vec![0.0, 0.0], vec![1.0, 0.0]],
                     vec![vec![0.0, 0.9], vec![1.0, 9.7], vec![0.0, 2.0], vec![0.0, 0.1], vec![1.0, 1.0]], 
                     vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 0.0], vec![0.0, 0.8], vec![1.0, 1.1]], 
                     vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 0.0], vec![0.0, 0.8], vec![1.0, 1.1]], 
                     vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 0.0], vec![0.0, 0.8], vec![1.0, 1.1]]]; 
    let w = vec![vec![vec![1.0, 0.0], vec![1.0, -1.0], vec![1.0, 0.0]],
                 vec![vec![1.0, -1.0], vec![1.0, 4.0], vec![1.0, -1.0]],
                 vec![vec![1.0, 0.0], vec![1.0, -1.0], vec![1.0, 0.0]]
                 ];
    let b = vec![1.0, 0.0];
    x = conv(x, &w, &b);
    x = conv(x, &w, &b);
    println!("img: {:?}", x);
    x = maxpool(x);
    let y = flatten(&x);
    println!("img: {:?}", x);
    println!("vec: {:?}", y);

    Ok(())
}

fn max(x: f64, y: f64) -> f64{
    if x > y{
        x
    } else{
        y
    }
}

fn maxpool(x: Vec<Vec<Vec<f64>>>) -> Vec<Vec<Vec<f64>>> {
    let mut res = vec![vec![vec![0.0; x[0][0].len()]; x[0].len() / 2]; x.len() / 2];
    for i in 0..x.len()-1{
        if i % 2 == 0{
            for j in 0..x[0].len()-1{
                if j % 2 == 0{
                    for k in 0..x[0][0].len(){
                        for k1 in 0..2{
                            let idi = i as i32 as i32+k1 as i32;
                            for k2 in 0..2{
                                let idj = j as i32 as i32+k2 as i32;
                                if idi < (x.len() as i32) && idj < (x[0].len() as i32){
                                    // println!("i: {}, j: {}, k: {}, w: {}, x: {}", idi, idj, k, w[k1][k2][k], x[idi as usize][idj as usize][k]);
                                    res[i/2][j/2][k] = max(res[i/2][j/2][k], x[idi as usize][idj as usize][k]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    res
}

fn flatten(x: &[Vec<Vec<f64>>]) -> Vec<f64>{
    let mut res = vec![0.0; x.len() * x[0].len() * x[0][0].len()];
    for i in 0..x.len(){
        for j in 0..x[0].len(){
            for k in 0..x[0][0].len(){
                res[i * x[0].len() * x[0][0].len() + j * x[0][0].len() + k] = x[i][j][k];
            }
        }
    }
    res
}

fn conv(x: Vec<Vec<Vec<f64>>>, w: &[Vec<Vec<f64>>], b: &[f64]) -> Vec<Vec<Vec<f64>>>{
    let ker_s = w.len();
    let disp = (ker_s - 1) / 2;
    let mut res = vec![vec![vec![0.0; x[0][0].len()]; x[0].len()]; x.len()];
    for i in 0..x.len(){
        for j in 0..x[0].len(){
            for k in 0..x[0][0].len(){
                for k1 in 0..w.len(){
                    let idi = i as i32-disp as i32+k1 as i32;
                    for k2 in 0..w[0].len(){
                        let idj = j as i32-disp as i32+k2 as i32;
                        if (idi >= 0) && idj >= 0 && idi < (x.len() as i32) && idj < (x[0].len() as i32){ 
                            // println!("i: {}, j: {}, k: {}, w: {}, x: {}", idi, idj, k, w[k1][k2][k], x[idi as usize][idj as usize][k]);
                            res[i][j][k] += x[idi as usize][idj as usize][k] * w[k1][k2][k];
                        }
                    }
                }
                res[i][j][k] += b[k];
                res[i][j][k] = relu(res[i][j][k]);
            }
        }
    }
    res
}

fn fc(x: Vec<f64>, w: &[Vec<f64>], b: &[f64]) -> Vec<f64>{
    let mut res = vec![0.0; b.len()];
    for i in 0..w.len(){
        for j in 0..w[0].len(){
            res[i] += x[j] * w[i][j];
        }
        res[i] = relu(res[i]+b[i]);
    }
    res
}

fn relu(x: f64) -> f64{
    if x >= 0.0 {
        x
    } else {
        0.0
    }
}

fn softmax(mut v: Vec<f64>) -> Vec<f64>{
    let mut sum = 0 as f64;
    for i in 0..v.len() {
        sum += v[i].exp();
    }
    for i in 0..v.len() {
        v[i] = v[i].exp() / sum;
    }
    v
}
