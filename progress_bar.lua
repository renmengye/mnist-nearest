local ProgressBar = {}

function ProgressBar.get(N)
    local progress = 0
    local finished = false
    return function(n)
        while n / N > progress / 80 do
           io.write('.')
           io.flush()
           progress = progress + 1
        end
        if progress == 80 and not finished then
            io.write('\n')
            io.flush()
            finished = true
        end
    end
end

return ProgressBar
