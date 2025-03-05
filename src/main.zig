const std = @import("std");
const mem = std.mem;
const Allocator = std.mem.Allocator;

fn BKDF(comptime Hash: type) type {
    const hash_len = Hash.digest_length;
    const key_len = Hash.block_length;
    const pepper_len_max = key_len;
    const version = 1;

    const Prf = struct {
        fn init(key: []const u8) Hash {
            std.debug.assert(key.len <= Hash.block_length);
            var h = Hash.init(.{});
            var prefix = [_]u8{0} ** Hash.block_length;
            @memcpy(prefix[0..key.len], key);
            h.update(key);
            return h;
        }
    };

    return struct {
        fn int(comptime T: type, n: T) [@sizeOf(T)]u8 {
            var t: [@sizeOf(T)]u8 = undefined;
            mem.writeInt(T, &t, n, .little);
            return t;
        }

        fn balloonCore(
            allocator: Allocator,
            key: [key_len]u8,
            personalization: []const u8,
            space_cost_log: u5,
            time_cost: u32,
            parallelism: u32,
            iteration: u32,
        ) ![hash_len]u8 {
            const space_cost = @as(usize, 1) << space_cost_log;
            var buffer = try allocator.alloc([hash_len]u8, space_cost);
            defer allocator.free(buffer);
            comptime std.debug.assert(hash_len % 4 == 0);
            const reps = (space_cost * time_cost * 3) / (hash_len / 4);
            var pseudorandom = try allocator.alloc(u8, space_cost * time_cost * 12);
            defer allocator.free(pseudorandom);

            const empty_key = [_]u8{0} ** key_len;
            var h = Prf.init(&empty_key);
            h.update(&int(u32, version));
            h.update(personalization);
            h.update(&int(u32, @intCast(space_cost)));
            h.update(&int(u32, time_cost));
            h.update(&int(u32, parallelism));
            h.update(&int(u32, iteration));
            for (0..reps) |i| {
                var h2 = h;
                h2.update(&int(u64, i));
                h2.final(pseudorandom[i * hash_len ..][0..hash_len]);
            }

            const hk = Prf.init(&key);
            h = hk;
            h.update(&int(u32, version));
            h.update(&int(u32, @intCast(space_cost)));
            h.update(&int(u32, time_cost));
            h.update(&int(u32, parallelism));
            h.update(&int(u32, iteration));
            h.update(&int(u64, reps));
            h.final(&buffer[0]);

            var counter = @as(u64, reps);
            for (1..space_cost) |i| {
                h = hk;
                h.update(&buffer[i - 1]);
                h.update(&int(u64, counter + i));
                h.final(&buffer[i]);
            }
            counter += space_cost;

            var offset: usize = 0;
            var previous = &buffer[space_cost - 1];
            const space_cost_mask = space_cost - 1;
            for (0..time_cost) |_| {
                for (0..space_cost) |m| {
                    const other1 = mem.readInt(u32, pseudorandom[offset..][0..4], .little) & space_cost_mask;
                    const other2 = mem.readInt(u32, pseudorandom[offset..][4..8], .little) & space_cost_mask;
                    const other3 = mem.readInt(u32, pseudorandom[offset..][8..12], .little) & space_cost_mask;
                    h = hk;
                    h.update(previous);
                    h.update(&buffer[m]);
                    h.update(&buffer[other1]);
                    h.update(&buffer[other2]);
                    h.update(&buffer[other3]);
                    h.update(&int(u64, counter));
                    counter += 1;
                    previous = &buffer[m];
                    h.final(previous);
                    offset += 12;
                }
            }
            return previous.*;
        }

        fn balloonCoreThreadWorker(
            allocator: Allocator,
            ok: *std.atomic.Value(bool),
            out: *[hash_len]u8,
            key: [key_len]u8,
            personalization: []const u8,
            space_cost_log: u5,
            time_cost: u32,
            parallelism: u32,
            iteration: u32,
        ) void {
            out.* = balloonCore(allocator, key, personalization, space_cost_log, time_cost, parallelism, iteration) catch {
                @branchHint(.unlikely);
                ok.store(false, .monotonic);
                return;
            };
        }

        pub fn kdf(
            allocator: Allocator,
            out: []u8,
            password: []const u8,
            salt: []const u8,
            personalization: []const u8,
            space_cost: u5,
            time_cost: u32,
            parallelism: u32,
            pepper: []const u8,
            associated_data: []const u8,
        ) !void {
            if (pepper.len > pepper_len_max) return error.InvalidParameter;

            var outputs = try allocator.alloc([hash_len]u8, parallelism);
            defer allocator.free(outputs);
            var key = [_]u8{0} ** key_len;
            @memcpy(key[0..pepper.len], pepper);
            var h = Prf.init(&key);
            h.update(password);
            h.update(salt);
            h.update(personalization);
            h.update(associated_data);
            h.update(&int(u32, @intCast(pepper.len)));
            h.update(&int(u32, @intCast(password.len)));
            h.update(&int(u32, @intCast(salt.len)));
            h.update(&int(u32, @intCast(personalization.len)));
            h.update(&int(u32, @intCast(associated_data.len)));
            h.final(key[0..hash_len]);
            if (key_len > hash_len) {
                @memset(key[hash_len..], 0);
            }

            if (parallelism == 1) {
                outputs[0] = try balloonCore(allocator, key, personalization, space_cost, time_cost, parallelism, 1);
            } else {
                var pool: std.Thread.Pool = undefined;
                try pool.init(.{ .allocator = allocator });
                defer pool.deinit();
                var ok = std.atomic.Value(bool).init(true);
                for (0..parallelism) |i| {
                    _ = try pool.spawn(balloonCoreThreadWorker, .{
                        allocator,
                        &ok,
                        &outputs[i],
                        key,
                        personalization,
                        space_cost,
                        time_cost,
                        parallelism,
                        @as(u32, @intCast(i + 1)),
                    });
                }
                if (!ok.load(.monotonic)) return error.OutOfMemory;
            }

            var hash = outputs[0];
            for (outputs[1..]) |output| {
                for (&hash, output) |*x, byte| x.* ^= byte;
            }

            const hk = Prf.init(&key);
            var previous = hash;
            for (0..out.len / hash_len) |i| {
                h = hk;
                h.update(&previous);
                h.update("bkdf");
                h.update(&int(u32, @intCast(i + 1)));
                h.final(out[i * hash_len ..][0..hash_len]);
            }
            const left = out.len % hash_len;
            if (left > 0) {
                var pad: [hash_len]u8 = undefined;
                h = hk;
                h.update(&previous);
                h.update("bkdf");
                h.update(&int(u32, @intCast(out.len / hash_len + 1)));
                h.final(&pad);
                @memcpy(out[out.len - left ..][0..left], pad[0..left]);
            }
        }
    };
}

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{ .safety = true }).init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    const Kdf = BKDF(std.crypto.hash.sha2.Sha256);
    const password = "password";
    const salt = "salt";
    const personalization = "";
    const pepper = "";
    const associated_data = "";
    var out: [32]u8 = undefined;
    try Kdf.kdf(
        allocator,
        &out,
        password,
        salt,
        personalization,
        1,
        1,
        1,
        pepper,
        associated_data,
    );
    std.debug.print("{x}", .{out});
}
