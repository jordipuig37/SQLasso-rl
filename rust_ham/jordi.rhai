if "tick" in memory == false {
        info("NOT SET");
        memory.tick = 0;
} else {
        memory.tick += 1;
}

fn execute_random(worker_idx) {
    let r = (rand() % 4).abs();
    switch r {
        0 => worker(worker_idx).move_up(),
        1 => worker(worker_idx).move_down(),
        2 => worker(worker_idx).move_right(),
        3 => worker(worker_idx).move_left(),
    }
}


for w in 0..8 {
    let x = 20 - worker(w).x;
    let y = 20 - worker(w).y;
    let r = (rand() % 40).abs();
    if r > x+y {
        execute_random(w)
    }
    else {
        if abs(x) > abs(y) {
            if x > 0 {
                worker(w).move_right();
            }
            else {
                worker(w).move_left();
            }
        }
        else {
            if y > 0 {
                worker(w).move_up();
            }
            else {
                worker(w).move_down();
                info("oponent tonto");
            }
        }
    }
    info(`worker ${w} finished`);
}
