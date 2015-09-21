local os = require('os')
local torch = require('torch')
local LoggerClass = torch.class('Logger')
local verboseThreshold = os.getenv('VERBOSE')
if verboseThreshold == nil then
    verboseThreshold = 0
else
    verboseThreshold = tonumber(verboseThreshold)
end

local term = {
    normal = '\027[0m',
    bright = '\027[1m',
    invert = '\027[7m',
    black = '\027[30m', 
    red = '\027[31m', 
    green = '\027[32m', 
    yellow = '\027[33m', 
    blue = '\027[34m', 
    magenta = '\027[35m',
    cyan = '\027[36m',
    white = '\027[37m', 
    default = '\027[39m'
    -- onblack = '\027[47m',
    -- onred = '\027[49m',
    -- ongreen = '\027[0m', 
    -- onyellow = '\027[1m',
    -- onblue = '\027[7m'
}
Logger.type = {
    INFO = 0,
    WARNING = 1,
    ERROR = 2,
    FATAL = 3
}

----------------------------------------------------------------------
function Logger:__init()
    --self.filename = filename
end

----------------------------------------------------------------------
function Logger.typeString(typ)
    if typ == Logger.type.INFO then
        return string.format('%sINFO:%s', term.green, term.default)
    elseif typ == Logger.type.WARNING then
        return string.format('%sWARNING:%s', term.yellow, term.default)
    elseif typ == Logger.type.ERROR then
        return string.format('%sERROR:%s', term.red, term.default)
    elseif typ == Logger.type.FATAL then
        return string.format('%sFATAL:%s', term.red, term.default)
    else
        return 'UNKNOWN'
    end
end

----------------------------------------------------------------------
function Logger.timeString(time)
    local date = os.date('*t', time)
    return string.format('%04d-%02d-%02d %02d:%02d:%02d',
        date.year, date.month, date.day, 
        date.hour, date.min, date.sec)
end

----------------------------------------------------------------------
function string.endswith(String, End)
    return End == '' or string.sub(String,-string.len(End)) == End
end

----------------------------------------------------------------------
function string.startswith(String, Start)
    return Start == '' or string.sub(String, 1, #Start) == Start
end

----------------------------------------------------------------------
function Logger:log(typ, text, verboseLevel)
    if verboseLevel == nil then
        verboseLevel = 0
    end
    if verboseLevel <= verboseThreshold then
        local info
        for i = 2,5 do
            info = debug.getinfo(i)
            if not string.endswith(info.short_src, 'logger.lua') then
                break
            end
        end
        local src = info.short_src
        if string.startswith(src, './') then
            src = string.sub(src, 3)
        end
        print(string.format(
            '%s %s %s:%d %s',
            self.typeString(typ),
            self.timeString(os.time()),
            src, info.currentline, text))
    end
end

----------------------------------------------------------------------
function Logger:logInfo(text, verboseLevel)
    if verboseLevel == nil then
        verboseLevel = 0
    end
    self:log(Logger.type.INFO, text, verboseLevel)
end

----------------------------------------------------------------------
function Logger:logWarning(text)
    self:log(Logger.type.WARNING, text, 0)
end

----------------------------------------------------------------------
function Logger:logError(text)
    self:log(Logger.type.ERROR, text, 0)
end

----------------------------------------------------------------------
function Logger:logFatal(text)
    self:log(Logger.type.FATAL, text, 0)
    os.exit(0)
end

----------------------------------------------------------------------
return Logger
